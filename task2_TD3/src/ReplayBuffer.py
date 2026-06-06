import torch
import torch.nn as nn
import torch.nn.functional as F
from task2_TD3.config.config import TD3config
from task1_decoder_only_transformer.config.transformer_config import TransformerConfig

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

    def add_old(self, state, action, reward, next_state, done):
        for i in range(self.n_envs):
            idx = self.curr_D_size % self.D_size
            self.states[idx]      = state[i]
            self.actions[idx]     = action[i]
            self.rewards[idx]     = reward[i]
            self.next_states[idx] = next_state[i]
            self.dones[idx]       = done[i]
            self.env_ids[idx]     = i
            self.curr_D_size += 1

    def add(self, state, action, reward, next_state, done):
        indices = torch.arange(self.n_envs) + self.curr_D_size
        indices = indices % self.D_size
        self.states[indices]      = state
        self.actions[indices]     = action
        self.rewards[indices]     = reward
        self.next_states[indices] = next_state
        self.dones[indices]       = done
        self.env_ids[indices]     = torch.arange(self.n_envs, device=self.device)
        self.curr_D_size += self.n_envs


    def sample(self, batch_size):
        indices = torch.randint(0, min(self.curr_D_size, self.D_size), (batch_size,), device=self.device)        
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

    def batch_get_history_old(self, b_idx):
        batch_states  = []
        batch_actions = []
        batch_masks   = []
        for idx in b_idx:
            s, a, m = self.get_history(idx.item())
            batch_states.append(s)
            batch_actions.append(a)
            batch_masks.append(m)   
        return torch.stack(batch_states), torch.stack(batch_actions), torch.stack(batch_masks)
    
    def batch_get_history(self, b_idx):
        """
        Vectorized history fetch. No Python loops over batch.
        b_idx: (B,) tensor of buffer indices
        Returns:
            states  : (B, L, obs_features)
            actions : (B, L, act_features)
            masks   : (B, L) bool — True = masked
        """
        B = b_idx.shape[0]
        L = self.L
        device = self.device

        # Build index matrix: (B, L) where each row is [idx, idx-1, ..., idx-(L-1)]
        offsets = torch.arange(L - 1, -1, -1, device=device).unsqueeze(0)  # (1, L)
        all_idx = (b_idx.unsqueeze(1) - offsets) % self.D_size              # (B, L)

        # Gather states, actions, dones, env_ids
        states_hist  = self.states[all_idx]   # (B, L, obs_features)
        actions_hist = self.actions[all_idx]  # (B, L, act_features)
        dones_hist   = self.dones[all_idx].squeeze(-1)    # (B, L)
        envid_hist   = self.env_ids[all_idx]              # (B, L)

        # Reference env_id is from the query index (rightmost = most recent)
        ref_env = self.env_ids[b_idx].unsqueeze(1)  # (B, 1)

        # Valid: same env_id AND not past an episode boundary
        # Walk left-to-right (oldest to newest): once a done or env mismatch is hit, all older are invalid
        # We work right-to-left: position L-1 is always valid if env matches
        env_match = (envid_hist == ref_env)  # (B, L)

        # Episode boundary: done at position i invalidates positions 0..i-1
        # Flip to right-to-left, cumsum to propagate invalidity
        dones_flipped = dones_hist.flip(dims=[1])          # (B, L) newest first
        # shift by 1: a done at step t invalidates steps before t
        boundary = torch.zeros_like(dones_flipped)
        boundary[:, 1:] = dones_flipped[:, :-1]
        invalid_from_done = boundary.cumsum(dim=1).flip(dims=[1]).bool()  # (B, L)

        valid = env_match & ~invalid_from_done  # (B, L)

        # Zero out invalid positions
        states_hist  = states_hist  * valid.unsqueeze(-1).float()
        actions_hist = actions_hist * valid.unsqueeze(-1).float()

        # Padding mask: True where invalid (padded)
        padding_mask = ~valid  # (B, L)

        return states_hist, actions_hist, padding_mask