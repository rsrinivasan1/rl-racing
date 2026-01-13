import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
import shutil
import csv
import time
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from collections import deque
from memory import PPOMemory
from network import Actor, Critic, SharedCNN
from config import batch_size, learning_rate, n_epochs, gamma, gae_lambda, c_1, c_2, eps, N, n_games, n_envs


def calculate_actor_loss(old_log_probs, log_probs, adv_batch):
    ratio = (log_probs - old_log_probs).exp()
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv_batch

    loss = torch.min(ratio * adv_batch, clipped).mean()
    return -loss


def choose_actions(states, actor, critic, action_map):
    """
    Choose actions for all environments based on the current states.
    
    Args:
        states: Stacked frames for all environments (shape: [n_envs, 4, height, width]).
        actor: The actor network (policy network) used to sample actions.
        critic: The critic network (value network) used to estimate state values.
        action_map: A dictionary mapping discrete actions to continuous actions.
    
    Returns:
        actions: A list of discrete actions sampled from the actor's policy for each environment.
        mapped_actions: A numpy array of continuous actions, mapped from the discrete actions using `action_map`.
        prob_actions: A list of log probabilities of the sampled actions for each environment.
        values: A list of state values estimated by the critic for each environment.
    """
    states = torch.tensor(states, dtype=torch.float).to(device)

    with torch.no_grad():
        distributions = actor(states)
        values = critic(states)

    actions = distributions.sample()
    prob_actions = distributions.log_prob(actions)

    actions = actions.tolist()
    # convert actions to continuous action using action_map
    mapped_actions = np.array([action_map[action] for action in actions])

    prob_actions = prob_actions.tolist()
    values = values.squeeze(-1).tolist()

    return actions, mapped_actions, prob_actions, values


def preprocess_states(states, frame_history):
    """
    Preprocess states and update frame history for all environments.
    Args:
        states: Current states from the environment (shape: [n_envs, height, width, channels]).
        frame_history: List of previous frames for each environment (shape: [n_envs, 4, height, width]).
    Returns:
        frame_history: Updated frame history.
    """
    # Crop out the bottom 12 pixels
    states = states[:, :-12, :, :]
    # Resize to 96x96
    states = np.array([cv2.resize(state, (96, 96)) for state in states])
    
    # Get pixels that are mostly red, mark them
    red_mask = (states[..., 0] > 100) & (states[..., 1] < 60) & (states[..., 2] < 60)
    states[red_mask] = 255

    # Mark border as grass
    states[states < 100] = 255

    # Convert to grayscale and add frame_width dimension at axis=1
    states = np.expand_dims(np.dot(states[..., :3], [0.2989, 0.5870, 0.1140]), axis=1)  # Shape: [n_envs, 1, height, width]    
    # Turn gray track to white, everything else to black
    track_mask = states < 150
    states[track_mask] = 255
    states[~track_mask] = 0

    # Normalize
    states = states / 255.0

    # If frame_history is None, initialize it with the current state repeated 4 times
    if frame_history is None:
        frame_history = np.repeat(states, 4, axis=1)  # Shape: [n_envs, 4, height, width]
    else:
        # Shift frame history and add the new state
        frame_history = np.roll(frame_history, shift=-1, axis=1)
        # remove second dimension from states
        frame_history[:, -1] = states.squeeze(axis=1)

    # Display all four frames
    # fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    # for i in range(4):
    #     axs[i].imshow(frame_history[0, i], cmap='gray')
    #     axs[i].axis('off')
    # plt.show()

    # Stacked frames are already in the correct shape: [n_envs, 4, height, width]
    return frame_history


def step(optim, loss):
    optim.zero_grad()
    loss.backward()
    optim.step()


def learn(actor, critic, optim, memory, lr, next_value):
    # 1. Update learning rate
    for param_group in optim.param_groups:
        param_group['lr'] = lr

    # 2. Get data from memory
    # Assumes these return numpy arrays of shape (n_envs, T, ...)
    s_arr, a_arr, p_arr, v_arr, r_arr, d_arr = memory.generate_batches(N)

    all_advantages = []
    all_returns = []

    # 3. Calculate GAE and Returns BEFORE the epoch loop
    # We do this per environment because sequences are contiguous there
    for j in range(n_envs):
        rewards = r_arr[j]
        values = v_arr[j]
        dones = d_arr[j]
        
        advantage = np.zeros(len(rewards), dtype=np.float32)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            # If t is the last step, we use the external 'next_value' passed to the function
            if t == len(rewards) - 1:
                # next_value is a tensor/array of shape (n_envs,), selecting j-th env
                next_val = next_value[j]
            else:
                next_val = values[t + 1]

            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            advantage[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        all_advantages.append(advantage)
        all_returns.append(advantage + values)

    # 4. Flatten all data (Combine all environments into one big buffer)
    # This is the "PPO way" - it treats all transitions as independent samples
    t_states = torch.tensor(np.concatenate(s_arr), dtype=torch.float).to(device)
    t_actions = torch.tensor(np.concatenate(a_arr), dtype=torch.long).to(device)
    t_old_probs = torch.tensor(np.concatenate(p_arr), dtype=torch.float).to(device)
    t_advantages = torch.tensor(np.concatenate(all_advantages), dtype=torch.float).to(device)
    t_returns = torch.tensor(np.concatenate(all_returns), dtype=torch.float).to(device)

    # 5. Normalize Advantages (Global normalization is more stable)
    t_advantages = (t_advantages - t_advantages.mean()) / (t_advantages.std() + 1e-8)

    # 6. The Learning Loop
    dataset_size = t_states.size(0)
    indices = np.arange(dataset_size)

    for _ in range(n_epochs):
        np.random.shuffle(indices)
        
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            idx = indices[start:end]

            # Mini-batch selection
            states = t_states[idx]
            actions = t_actions[idx]
            old_log_probs = t_old_probs[idx]
            advantages = t_advantages[idx]
            returns = t_returns[idx]

            # Forward pass
            distribution = actor(states)
            critic_values = critic(states).squeeze()
            new_log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy().mean()

            # Policy Loss (PPO Clip)
            actor_loss = calculate_actor_loss(old_log_probs, new_log_probs, advantages)

            # Value Loss (MSE between critic prediction and TD-lambda returns)
            critic_loss = (returns - critic_values).pow(2).mean()

            # Total Loss
            total_loss = actor_loss + c_1 * critic_loss - c_2 * entropy

            # Optimizer step
            optim.zero_grad()
            total_loss.backward()
            optim.step()

    memory.clear_memory()


def run(envs, actor, critic, memory, checkpoint_file, record, anneal_lr=True):
    best_score = -float('inf')
    best_mean_score = -float('inf')
    prev_scores = []
    num_steps = 0

    action_map = {
        0: [-1, 0, 0],   # turn left
        1: [1, 0, 0],    # turn right
        2: [0, 1, 0],    # accelerate
        3: [0, 0, 0.8],  # brake
        4: [0, 0, 0]     # do nothing
    }

    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        start_episode = checkpoint['episode']
        best_score = checkpoint['best_score']
        best_mean_score = checkpoint['best_mean_score']
        prev_scores = checkpoint['prev_scores']
        print(f"Loaded checkpoint from {checkpoint_file}, starting from episode {start_episode}")
    else:
        start_episode = 0

    optim = torch.optim.Adam(
        set(actor.parameters()) | set(critic.parameters()), lr=learning_rate, eps=1e-5
    )

    # initialize frame history for each env
    frame_history = None
    lr = learning_rate
    if anneal_lr:
        min_lr = 0.0001
        frac = 1 - (start_episode / n_games)
        lr = max(min_lr, learning_rate * frac)

    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    log_file = "./logs/mean_scores.csv"
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Mean Score"])

    # want to learn every N games
    for i in tqdm(range(start_episode, n_games), desc="Training episodes"):
        states = envs.reset()[0]
        done = False
        scores = np.zeros(n_envs)

        repeat_num = 4

        if record:
            envs.envs[0].start_recording("current")
        
        while not done:
            states = preprocess_states(states, frame_history)
            frame_history = states
            actions, mapped_actions, probs, vals = choose_actions(states, actor, critic, action_map)

            total_rewards = np.zeros(n_envs)
            dones_received = np.zeros(n_envs, dtype=bool)
            mask = ~dones_received
            
            for _ in range(repeat_num):
                # repeat action
                next_states, rewards, terminated, truncated, _ = envs.step(mapped_actions)
                dones_received = dones_received | terminated | truncated

                # clip to avoid incentivizing going too fast and hitting two tiles
                total_rewards[mask] += np.clip(rewards[mask], 0, 1)

                reset_history_for_done_frames(frame_history, next_states, dones_received)
                if done := all(dones_received):
                    break
            
            num_steps += 1
            scores += total_rewards

            # store this observation
            memory.store_memory(states, actions, probs, vals, total_rewards, dones_received)

            if num_steps % N == 0:
                # Get next state values for GAE
                next_states_processed = preprocess_states(next_states, frame_history.copy())
                with torch.no_grad():
                    next_values = critic(torch.tensor(next_states_processed, dtype=torch.float).to(device)).cpu().numpy()
                
                # anneal learning rate if specified
                if anneal_lr:
                    min_lr = 0.00001
                    frac = 1 - (i / n_games)
                    lr = max(min_lr, learning_rate * frac)
                
                # actually backpropagate with next values
                learn(actor, critic, optim, memory, lr, next_values)

            states = next_states
        
        if scores[0] > best_score:
            best_score = scores[0]
            # save recording
            if record:
                tqdm.write(f"Best score, saving recording")
                shutil.rmtree("./videos")
                os.makedirs("./videos")
                envs.envs[0].stop_recording()  # Close the current recording
                shutil.move("./videos/current.mp4", f"./videos/best_{int(best_score)}.mp4")
        
        # average score over all envs
        score = np.mean(scores)
        prev_scores.append(score)
        mean_score = np.mean(prev_scores[-100:])
        
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([i, mean_score])

        tqdm.write(f"Episode {i}, lr: {round(lr, 5)}, score: {score}, mean score: {mean_score}")
        if mean_score > best_mean_score:
            best_mean_score = mean_score
            tqdm.write(f"Best average score over 100 trials: {best_mean_score}")
        
        # Save model weights every 50 episodes
        if (i + 1) % 50 == 0:
            checkpoint_path = f"./checkpoints/checkpoint_episode_{i + 1}.pth"
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'episode': i + 1,
                'best_score': best_score,
                'best_mean_score': best_mean_score,
                'prev_scores': prev_scores,
            }, checkpoint_path)
            tqdm.write(f"Saved checkpoint at episode {i + 1} to {checkpoint_path}")

    envs.close()


def preprocess_state_single_env(state):
    # Crop out the bottom 12 pixels
    state = state[:-12, :, :]
    # Resize to 96x96
    state = cv2.resize(state, (96, 96))
    # Convert to grayscale
    state = np.dot(state[..., :3], [0.2989, 0.5870, 0.1140])
    # Turn gray track to black, everything else to white
    state[state < 150] = 0
    state[state >= 150] = 255
    # Add frame_width dimension at axis=0
    state = np.expand_dims(state, axis=0)  # Shape: [1, height, width]
    # Repeat the frame 4 times to create the stacked frame
    state = np.repeat(state, 4, axis=0)  # Shape: [4, height, width]
    return state


def reset_history_for_done_frames(frame_history, next_states, dones):
    for i, done in enumerate(dones):
        if done:
            frame_history[i] = preprocess_state_single_env(next_states[i])


def make_env(gym_id, record_video=False, video_folder='./videos'):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array")
        if record_video:
            env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: False)
        return env
    return thunk


def plot_rewards():
    """Plot rewards over time from the training log."""
    log_file = "./logs/mean_scores.csv"
    if not os.path.exists(log_file):
        print("No training log found. Run training first.")
        return
    
    import pandas as pd
    
    # Read the CSV file
    df = pd.read_csv(log_file)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['Episode'], df['Mean Score'], linewidth=2, alpha=0.8)
    plt.title('Training Progress: Mean Score Over Episodes', fontsize=16)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Mean Score (100-episode average)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    max_score = df['Mean Score'].max()
    max_episode = df.loc[df['Mean Score'].idxmax(), 'Episode']
    plt.axhline(y=max_score, color='r', linestyle='--', alpha=0.7, label=f'Best: {max_score:.1f} (Episode {max_episode})')
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    os.makedirs("./plots", exist_ok=True)
    plt.savefig('./plots/training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to ./plots/training_progress.png")


if __name__ == "__main__":
    envs = gym.vector.SyncVectorEnv([make_env('CarRacing-v3', record_video=(i == 0)) for i in range(n_envs)])
    checkpoint_file = input("Enter checkpoint file (leave empty for none): ")
    device = torch.device("cpu")

    shared_cnn = SharedCNN().to(device)
    actor = Actor(shared_cnn, device)
    critic = Critic(shared_cnn, device)

    memory = PPOMemory(batch_size, n_envs)

    start = time.time()
    run(envs, actor, critic, memory, checkpoint_file, True, anneal_lr=True)
    print(f"Training took {(time.time() - start) // 60} min")

    # Build graph of rewards over time
    plot_rewards()