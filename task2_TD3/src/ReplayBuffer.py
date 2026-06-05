import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import TD3config, TransformerConfig


class ReplayBuffer():
    def __init__(self, config: TD3config, tConfig: TransformerConfig, device):
        self.device = device
        self.obs_features = config.obs_features
        self.act_features = config.act_features
        self.states      = torch.zeros(config.D_size, config.obs_features, device=device)
        self.actions     = torch.zeros(config.D_size, config.act_features, device=device)
        self.rewards     = torch.zeros(config.D_size, 1, device=device)
        self.next_states = torch.zeros(config.D_size, config.obs_features, device=device)
        self.dones       = torch.zeros(config.D_size, 1, device=device)
        self.env_ids     = torch.zeros(config.D_size, dtype=torch.long, device=device)
        self.curr_D_size = 0
        self.n_envs      = config.n_envs
        self.D_size      = config.D_size
        self.L           = tConfig.context_window

    def add(self, state, action, reward, next_state, done):
        for i in range(self.n_envs):
            idx = self.curr_D_size % self.D_size
            self.states[idx]      = state[i]
            self.actions[idx]     = action[i]
            self.rewards[idx]     = reward[i]
            self.next_states[idx] = next_state[i]
            self.dones[idx]       = done[i]
            self.env_ids[idx]     = i
            self.curr_D_size += 1

    def sample(self, batch_size):
        indices = torch.randint(0, min(self.curr_D_size, self.D_size), (batch_size,))
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices, ##added indices to get history before a state
        )

    def get_current_history(self,env_id):
        #we have the current_size of D with (S,A,R,S_,d) as last element S_ is the last element. Thus idx with current_size+1
        idx = self.curr_D_size - 1
        history_states  = []
        history_actions = []
        
        for i in range(self.L):
            curr_idx = (idx - i) % self.D_size
            if self.env_ids[curr_idx].item() != env_id:
                continue
            history_states.append(self.states[curr_idx])
            history_actions.append(self.actions[curr_idx])
            if i > 0 and self.dones[curr_idx].item() == 1:
                break

        history_states  = history_states[::-1]
        history_actions = history_actions[::-1]

        valid_len = len(history_states)
        pad_len   = self.L - valid_len

        state_pad  = torch.zeros(pad_len, self.obs_features, device=self.device)
        action_pad = torch.zeros(pad_len, self.act_features, device=self.device)

        if valid_len > 0:
            states_tensor  = torch.cat([state_pad,  torch.stack(history_states)],  dim=0)
            actions_tensor = torch.cat([action_pad, torch.stack(history_actions)], dim=0)
        else:
            states_tensor  = state_pad
            actions_tensor = action_pad

        padding_mask = torch.ones(self.L, dtype=torch.bool, device=self.device)
        padding_mask[pad_len:] = False

        return states_tensor, actions_tensor, padding_mask

    def get_history(self, idx):
        env_id = self.env_ids[idx].item()
        history_states  = []
        history_actions = []

        for i in range(self.L):
            curr_idx = (idx - i) % self.D_size
            if self.env_ids[curr_idx].item() != env_id:
                continue
            history_states.append(self.states[curr_idx])
            history_actions.append(self.actions[curr_idx])
            if i > 0 and self.dones[curr_idx].item() == 1:
                break

        history_states  = history_states[::-1]
        history_actions = history_actions[::-1]

        valid_len = len(history_states)
        pad_len   = self.L - valid_len

        state_pad  = torch.zeros(pad_len, self.obs_features, device=self.device)
        action_pad = torch.zeros(pad_len, self.act_features, device=self.device)

        if valid_len > 0:
            states_tensor  = torch.cat([state_pad,  torch.stack(history_states)],  dim=0)
            actions_tensor = torch.cat([action_pad, torch.stack(history_actions)], dim=0)
        else:
            states_tensor  = state_pad
            actions_tensor = action_pad

        padding_mask = torch.ones(self.L, dtype=torch.bool, device=self.device)
        padding_mask[pad_len:] = False

        return states_tensor, actions_tensor, padding_mask

    def batch_get_history(self, b_idx):
        batch_states  = []
        batch_actions = []
        batch_masks   = []
        for idx in b_idx:
            s, a, m = self.get_history(idx.item())
            batch_states.append(s)
            batch_actions.append(a)
            batch_masks.append(m)   
        return torch.stack(batch_states), torch.stack(batch_actions), torch.stack(batch_masks)