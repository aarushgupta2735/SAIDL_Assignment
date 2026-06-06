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
        self.curr_D_size    = 0   # capped at D_size, used for sampling
        self.curr_write_ptr = 0   # always advances mod D_size
        self.n_envs      = config.n_envs
        self.D_size      = config.D_size
        self.L           = tConfig.context_window

    def add(self, state, action, reward, next_state, done):
        indices = (torch.arange(self.n_envs, device=self.device) + self.curr_write_ptr) % self.D_size
        self.states[indices]      = state
        self.actions[indices]     = action
        self.rewards[indices]     = reward
        self.next_states[indices] = next_state
        self.dones[indices]       = done
        self.env_ids[indices]     = torch.arange(self.n_envs, device=self.device)
        self.curr_write_ptr = (self.curr_write_ptr + self.n_envs) % self.D_size
        self.curr_D_size = min(self.curr_D_size + self.n_envs, self.D_size)

    def sample(self, batch_size):
        indices = torch.randint(0, self.curr_D_size, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
        )

    def get_current_history(self, env_id):
        idx = (self.curr_write_ptr - 1) % self.D_size
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
        B = b_idx.shape[0]
        L = self.L
        device = self.device

        offsets = torch.arange(L - 1, -1, -1, device=device).unsqueeze(0)  # (1, L)
        all_idx = (b_idx.unsqueeze(1) - offsets) % self.D_size              # (B, L)

        states_hist  = self.states[all_idx]                    # (B, L, obs)
        actions_hist = self.actions[all_idx]                   # (B, L, act)
        dones_hist   = self.dones[all_idx].squeeze(-1)         # (B, L)
        envid_hist   = self.env_ids[all_idx]                   # (B, L)

        ref_env  = self.env_ids[b_idx].unsqueeze(1)            # (B, 1)
        env_match = (envid_hist == ref_env)                    # (B, L)

        # done at position i means episode ended there — positions before it in this window
        # belong to a previous episode. Propagate invalidity leftward.
        # all_idx is ordered oldest→newest (left→right), so we flip to newest→oldest,
        # shift done signal by 1 (done at step t invalidates steps before t), cumsum, flip back.
        dones_flipped = dones_hist.flip(dims=[1])              # newest first
        boundary = torch.zeros_like(dones_flipped)
        boundary[:, 1:] = dones_flipped[:, :-1]               # shift: done propagates left
        invalid_from_done = boundary.cumsum(dim=1).flip(dims=[1]).bool()  # (B, L)

        valid = env_match & ~invalid_from_done                 # (B, L)

        states_hist  = states_hist  * valid.unsqueeze(-1).float()
        actions_hist = actions_hist * valid.unsqueeze(-1).float()

        padding_mask = ~valid                                  # True = masked

        return states_hist, actions_hist, padding_mask