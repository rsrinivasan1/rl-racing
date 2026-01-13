import numpy as np

class PPOMemory:
    def __init__(self, batch_size, n_envs):
        self.states = [[] for _ in range(n_envs)]
        self.probs = [[] for _ in range(n_envs)]
        self.values = [[] for _ in range(n_envs)]
        self.actions = [[] for _ in range(n_envs)]
        self.rewards = [[] for _ in range(n_envs)]
        self.dones = [[] for _ in range(n_envs)]

        self.batch_size = batch_size
        self.n_envs = n_envs
    
    def generate_batches(self, n_states):
        return np.array(self.states), \
                np.array(self.actions),\
                np.array(self.probs), \
                np.array(self.values), \
                np.array(self.rewards), \
                np.array(self.dones)

    def store_memory(self, states, actions, probs, vals, rewards, dones):
        for env_idx in range(len(states)):
            self.states[env_idx].append(states[env_idx])
            self.probs[env_idx].append(probs[env_idx])
            self.values[env_idx].append(vals[env_idx])
            self.actions[env_idx].append(actions[env_idx])
            self.rewards[env_idx].append(rewards[env_idx])
            self.dones[env_idx].append(dones[env_idx])

    def clear_memory(self):
        self.states = [[] for _ in range(self.n_envs)]
        self.probs = [[] for _ in range(self.n_envs)]
        self.values = [[] for _ in range(self.n_envs)]
        self.actions = [[] for _ in range(self.n_envs)]
        self.rewards = [[] for _ in range(self.n_envs)]
        self.dones = [[] for _ in range(self.n_envs)]