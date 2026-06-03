import torch
import torch.nn as nn
import torch.nn.functional as F
from TD3.config.config import TD3config

class ReplayBuffer():
    def __init__(self,config:TD3config,device):
        self.states = torch.zeros(config.D_size, config.obs_features,device=device)
        self.actions = torch.zeros(config.D_size, config.act_features,device=device)
        self.rewards = torch.zeros(config.D_size, 1,device=device)
        self.next_states = torch.zeros(config.D_size, config.obs_features,device=device)
        self.dones = torch.zeros(config.D_size, 1,device=device)
        self.curr_D_size = 0 
        self.D_size = config.D_size

    def sample(self,batch_size):
        indices = torch.randint(0, self.curr_D_size, (batch_size,))
        return (self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], self.dones[indices])

    def add(self,state, action, reward, next_state, done):
        if(self.curr_D_size<self.D_size):
            self.states[self.curr_D_size] = state
            self.actions[self.curr_D_size] = action
            self.rewards[self.curr_D_size] = reward
            self.next_states[self.curr_D_size] = next_state
            self.dones[self.curr_D_size] = done
        else:
            self.states[self.curr_D_size % self.D_size] = state
            self.actions[self.curr_D_size % self.D_size] = action
            self.rewards[self.curr_D_size % self.D_size] = reward
            self.next_states[self.curr_D_size % self.D_size] = next_state
            self.dones[self.curr_D_size % self.D_size] = done
        self.curr_D_size += 1