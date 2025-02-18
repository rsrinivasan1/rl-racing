import gymnasium as gym
import time
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from memory import PPOMemory
from network import Actor, Critic, SharedCNN
from config import batch_size, learning_rate, n_epochs, gamma, gae_lambda, c_1, eps, N, n_games, n_envs


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
    # Convert to grayscale and add frame_width dimension at axis=1
    states = np.expand_dims(np.dot(states[..., :3], [0.2989, 0.5870, 0.1140]), axis=1)  # Shape: [n_envs, 1, height, width]
    # Turn gray track to black, everything else to white
    states[states < 150] = 0
    states[states >= 150] = 255

    # If frame_history is None, initialize it with the current state repeated 4 times
    if frame_history is None:
        frame_history = np.repeat(states, 4, axis=1)  # Shape: [n_envs, 4, height, width]
    else:
        # Shift frame history and add the new state
        frame_history = np.roll(frame_history, shift=-1, axis=1)
        # remove second dimension from states
        frame_history[:, -1] = states.squeeze(axis=1)

    # Stacked frames are already in the correct shape: [n_envs, 4, height, width]
    return frame_history


def step(actor_optim, critic_optim, loss):
    actor_optim.zero_grad()
    critic_optim.zero_grad()
    loss.backward()
    actor_optim.step()
    critic_optim.step()


# assumes you have N observations in memory, for each batch makes a step
def learn(actor, critic, actor_optim, critic_optim, memory, lr):
    # update lr for both optimizers:
    actor_optim.param_groups[0]['lr'] = lr
    critic_optim.param_groups[0]['lr'] = lr
    
    for i in range(n_epochs):
        # create batches from stored memory, shuffled each epoch
        states_arr, actions_arr, old_probs_arr, values_arr, rewards_arr, dones_arr, batches = memory.generate_batches(n_states=N)
        for j in range(n_envs):
            # calculate advantage for each env, for every state in memory
            advantage = np.zeros_like(rewards_arr[j])
            # get each A_t BEFORE using shuffled batches, so that continuity of states is not broken
            for t in range(len(rewards_arr[j]) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards_arr[j]) - 1):
                    # discount = (gamma * gae_lambda) ^ (k - t)
                    # A_t = sum of discount * (r_t + gamma * V(s_t+1) * (1 - done_t))
                    # no more extra rewards if done, just discount * rewards_arr[k] - values[k]
                    a_t += discount * (rewards_arr[j][k] + gamma * values_arr[j][k + 1] * (1 - int(dones_arr[j][k])) - values_arr[j][k])
                    discount *= gamma * gae_lambda
                    if dones_arr[j][k]:
                        # reset discount to 1 if episode ends
                        discount = 1
                advantage[t] = a_t

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
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value).pow(2).mean()

                total_loss = actor_loss + c_1 * critic_loss

                step(actor_optim, critic_optim, total_loss)

    memory.clear_memory()


def run(envs, actor, critic, actor_optim, critic_optim, memory, device, anneal_lr=True):
    best_score = -float('inf')
    prev_scores = []
    num_steps = 0

    action_map = {
        0: [-1, 0, 0],
        1: [1, 0, 0],
        2: [0, 1, 0.8],
        3: [0, 0, 0.8],
        4: [0, 0, 0]
    }

    # initialize frame history for each env
    frame_history = None

    # want to learn every N games
    for i in tqdm(range(n_games), desc="Training episodes"):
        states = envs.reset()[0]
        done = False
        scores = np.zeros(n_envs)
        lr = learning_rate
        while not done:
            states = preprocess_states(states, frame_history)
            actions, mapped_actions, probs, vals = choose_actions(states, actor, critic, action_map)

            active_envs = np.ones(n_envs, dtype=bool)  # All environments start as active
            for _ in range(20):
                # repeat action 20 times
                next_states, rewards, terminated, truncated, _ = envs.step(mapped_actions)
                dones = terminated | truncated
                if (done := all(dones)):
                    break
            num_steps += 1
            scores += rewards
            # store this observation
            memory.store_memory(states, actions, probs, vals, rewards, dones)

            if num_steps % N == 0:
                # anneal learning rate if specified
                if anneal_lr:
                    frac = 1 - (i / n_games)
                    lr = learning_rate * frac
                # actually backpropagate
                learn(actor, critic, actor_optim, critic_optim, memory, lr)
            states = next_states
        
        # average score over all envs
        score = np.mean(scores)
        prev_scores.append(score)
        mean_score = np.mean(prev_scores[-100:])

        print(f"Episode {i}, lr: {round(lr, 5)}, score: {score}, mean score: {mean_score}\n")
        if mean_score > best_score:
            best_score = mean_score
            print(f"Best average score over 100 trials: {best_score}\n")

    envs.close()


def make_env(gym_id):
    def thunk():
        return gym.make(gym_id)
    return thunk


if __name__ == "__main__":
    envs = gym.vector.SyncVectorEnv([make_env('CarRacing-v3') for _ in range(n_envs)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # outdim is 5 for the following:
    # 1. turn left
    # 2. turn right
    # 3. accelerate
    # 4. brake
    # 5. do nothing

    shared_cnn = SharedCNN().to(device)
    actor = Actor(shared_cnn, device)
    critic = Critic(shared_cnn, device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=learning_rate, eps=1e-5)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=learning_rate, eps=1e-5)

    memory = PPOMemory(batch_size, n_envs)

    start = time.time()
    run(envs, actor, critic, actor_optim, critic_optim, memory, device)
    print(f"Training took {(time.time() - start)} seconds")