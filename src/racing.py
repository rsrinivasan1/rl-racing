import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
import shutil
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
    
    # Get pixels that are mostly red
    red_mask = (states[..., 0] > 100) & (states[..., 1] < 60) & (states[..., 2] < 60)
    states[red_mask] = 255

    # Convert to grayscale and add frame_width dimension at axis=1
    states = np.expand_dims(np.dot(states[..., :3], [0.2989, 0.5870, 0.1140]), axis=1)  # Shape: [n_envs, 1, height, width]    
    # Turn gray track to black, everything else to white
    states[states < 150] = 0    
    states[states > 180] = 255

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


# assumes you have N observations in memory, for each batch makes a step
def learn(actor, critic, optim, memory, lr):
    # update lr for both optimizers:
    optim.param_groups[0]['lr'] = lr
    
    for i in range(n_epochs):
        # create batches from stored memory, shuffled each epoch
        states_arr, actions_arr, old_probs_arr, values_arr, rewards_arr, dones_arr, batches = memory.generate_batches(n_states=N)
        for j in range(n_envs):
            # calculate advantage for each env, for every state in memory
            advantage = np.zeros_like(rewards_arr[j])
            deltas = rewards_arr[j][:-1] + gamma * values_arr[j][1:] * (1 - dones_arr[j][:-1]) - values_arr[j][:-1]
            
            # compute GAE in a vectorized O(n) manner
            advantage[-1] = deltas[-1]  # last step advantage is just delta
            for t in reversed(range(len(deltas) - 1)):
                advantage[t] = deltas[t] + gamma * gae_lambda * (1 - dones_arr[j][t]) * advantage[t + 1]
            
            advantage = torch.tensor(advantage).to(device)
            values = torch.tensor(values_arr[j]).to(device)

            for batch in batches:
                states = torch.tensor(states_arr[j][batch], dtype=torch.float).to(device)
                old_probs = torch.tensor(old_probs_arr[j][batch], dtype=torch.float).to(device)
                actions = torch.tensor(actions_arr[j][batch], dtype=torch.long).to(device)

                distribution = actor(states)
                critic_value = critic(states)

                new_probs = distribution.log_prob(actions)
                
                actor_loss = calculate_actor_loss(old_probs, new_probs, advantage[batch])

                # total predicted reward of the state = advantage + value
                # newvalue = critic_value
                # b_returns = returns
                # b_values = values[batch]
                
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value).pow(2).mean()

                # entropy loss
                entropy = distribution.entropy().mean()
                total_loss = actor_loss + c_1 * critic_loss - c_2 * entropy
                step(optim, total_loss)

    memory.clear_memory()


def run(envs, actor, critic, optim, memory, device, anneal_lr=True):
    best_score = -float('inf')
    best_mean_score = -float('inf')
    prev_scores = []
    num_steps = 0

    action_map = {
        0: [-1, 0, 0],  # turn left
        1: [1, 0, 0],  # turn right
        2: [0, 1, 0],  # accelerate
        3: [0, 0, 0.8],  # brake
        4: [0, 0, 0]  # do nothing
    }

    # initialize frame history for each env
    frame_history = None

    # want to learn every N games
    for i in tqdm(range(n_games), desc="Training episodes"):
        states = envs.reset()[0]
        done = False
        scores = np.zeros(n_envs)
        lr = learning_rate

        repeat_num = 4
        history_len = 200 // repeat_num
        early_stopping_penalty = 100
        reward_history = [deque(maxlen=history_len) for _ in range(n_envs)]

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

                total_rewards[mask] += rewards[mask]

                reset_history_for_done_frames(frame_history, next_states, dones_received)
                if done := all(dones_received):
                    break
            # clip to avoid:
            # 1. -100 reward on hitting border
            # 2. incentivizing going too fast and hitting two tiles
            total_rewards[mask] = np.clip(total_rewards[mask], -0.1 * repeat_num, 0.9 * repeat_num)
            # print(f"Action: {actions[0]}, Reward: {total_rewards[0]}")
            
            # # if any of the envs aren't getting rewards, stop all
            # early_stop = False
            # for j, env in enumerate(reward_history):
            #     reward_history[j].append(total_rewards[j])
            #     if len(env) == history_len and np.mean(env) < -0.09 * repeat_num:
            #         early_stop = True
            #         # penalize the agent for not getting rewards
            #         total_rewards[j] -= early_stopping_penalty
            #         break
            
            num_steps += 1
            scores += total_rewards

            # store this observation
            memory.store_memory(states, actions, probs, vals, total_rewards, dones_received)
            
            # if early_stop:
            #     print("Early stopping")
            #     break

            if num_steps % N == 0:
                # anneal learning rate if specified
                if anneal_lr:
                    min_lr = 0.00001
                    frac = 1 - (i / n_games)
                    lr = max(min_lr, learning_rate * frac)
                # actually backpropagate
                learn(actor, critic, optim, memory, lr)
            states = next_states
        
        if scores[0] > best_score:
            best_score = scores[0]
            # save recording
            tqdm.write(f"Best score, saving recording")
            shutil.rmtree("./videos")
            os.makedirs("./videos")
            envs.envs[0].stop_recording()  # Close the current recording
            shutil.move("./videos/current.mp4", f"./videos/best_{int(best_score)}.mp4")
        # average score over all envs
        score = np.mean(scores)
        prev_scores.append(score)
        mean_score = np.mean(prev_scores[-100:])

        tqdm.write(f"Episode {i}, lr: {round(lr, 5)}, score: {score}, mean score: {mean_score}")
        if mean_score > best_mean_score:
            best_mean_score = mean_score
            tqdm.write(f"Best average score over 100 trials: {best_mean_score}")

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


if __name__ == "__main__":
    envs = gym.vector.SyncVectorEnv([make_env('CarRacing-v3', record_video=(i == 0)) for i in range(n_envs)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shared_cnn = SharedCNN().to(device)
    actor = Actor(shared_cnn, device)
    critic = Critic(shared_cnn, device)

    optim = optimizer = torch.optim.Adam(
        set(actor.parameters()) | set(critic.parameters()), lr=learning_rate, eps=1e-5
    )

    memory = PPOMemory(batch_size, n_envs)

    start = time.time()
    run(envs, actor, critic, optim, memory, device)
    print(f"Training took {(time.time() - start)} seconds")

    # TODO: 
    # Save recordings of the agent playing the game
    # Build graph of rewards over time