import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import TD3config,TransformerConfig

class ReplayBuffer():
    def __init__(self,config:TD3config,tConfig:TransformerConfig,device):
        self.states = torch.zeros(config.D_size, config.obs_features,device=device)
        self.actions = torch.zeros(config.D_size, config.act_features,device=device)
        self.rewards = torch.zeros(config.D_size, 1,device=device)
        self.next_states = torch.zeros(config.D_size, config.obs_features,device=device)
        self.dones = torch.zeros(config.D_size, 1,device=device)
        self.curr_D_size = 0 
        self.n_envs = config.n_envs
        self.D_size = config.D_size
        self.L = tConfig.context_window

    def sample(self,batch_size):
        indices = torch.randint(0, min(self.curr_D_size,self.D_size), (batch_size,))
        return (self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], self.dones[indices])

    def add(self, state, action, reward, next_state, done):
        for i in self.n_envs:
            idx = self.curr_D_size % self.D_size
            self.states[idx]      = state[i]
            self.actions[idx]     = action[i]
            self.rewards[idx]     = reward[i]
            self.next_states[idx] = next_state[i]
            self.dones[idx]       = done[i]
        self.curr_D_size += 1

    def get_history(self,idx):
        #returns last L pairs of states and actions for a given idx
        history_states = []
        history_actions = []
        padding_mask = torch.zeros((self.L,)).bool()
        for i in range(self.L):
            curr_idx = (idx-i)%self.D_size
            history_states.append(self.states[curr_idx])
            history_actions.append(self.actions[curr_idx])
            padding_mask[i] = 1
            if(self.dones[curr_idx==1]):
                break
        #add zeros
        history_actions+=history_actions+[0 for i in range(0,self.L-len(history_actions))]
        history_states+=history_states+[0 for i in range(0,self.L-len(history_states))]
        history_actions = history_actions[::-1]
        history_states = history_states[::-1]
        ##NEED A PADDIGN MASK
        return torch.tensor(history_states),torch.tensor(history_actions),padding_mask