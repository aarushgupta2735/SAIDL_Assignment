import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import platform
from task1_decoder_only_transformer.config.transformer_config import TransformerConfig
from task2_TD3.config.config import TD3config
from task2_TD3.src.Actor import Actor
from task2_TD3.src.Critic import Critic
from task2_TD3.src.ReplayBuffer import ReplayBuffer


class TD3(nn.Module):
    def __init__(self, config: TD3config, tConfig: TransformerConfig, buffer, device):
        super().__init__()
        self.device = device
        self.O = config.obs_features
        self.A = config.act_features
        self.B = config.batch_size
        self.D_size = config.D_size
        self.gamma = config.gamma
        self.exp_noise = config.exploration_noise
        self.policy_noise = config.policy_noise
        self.noise_clip = config.noise_clip
        self.a_low = config.a_low
        self.a_high = config.a_high
        self.actor_is_transformer = config.use_transformer
        self.n_envs = config.n_envs
        self.on_RLHF = config.on_RLHF

        self.theta = Actor(config, tConfig, device=device)
        self.theta_target = copy.deepcopy(self.theta)
        for param in self.theta_target.parameters():
            param.requires_grad = False

        self.phi1 = Critic(config, device=device)
        self.phi1_target = copy.deepcopy(self.phi1)
        for param in self.phi1_target.parameters():
            param.requires_grad = False

        self.phi2 = Critic(config, device=device)
        self.phi2_target = copy.deepcopy(self.phi2)
        for param in self.phi2_target.parameters():
            param.requires_grad = False

        if config.use_compile and platform.system() != "Windows":
            self.theta = torch.compile(self.theta)
            self.phi1  = torch.compile(self.phi1)
            self.phi2  = torch.compile(self.phi2)

        self.D = buffer

        self.phi1_optimiser = torch.optim.Adam(self.phi1.parameters(), lr=config.lr)
        self.phi2_optimiser = torch.optim.Adam(self.phi2.parameters(), lr=config.lr)
        self.actor_optimiser = torch.optim.Adam(self.theta.parameters(), lr=config.lr)

        self.polyak_coeff = config.polyak_coeff
        self.policy_delay = config.policy_delay

    def select_action(self, curr_state, explore=True, n_envs_override=None):
        n = self.n_envs if n_envs_override is None else n_envs_override
        if self.actor_is_transformer:
            batch_states  = []
            batch_actions = []
            batch_masks   = []
            for i in range(n):
                states, actions, masks = self.D.get_current_history(i)
                batch_states.append(states)
                batch_actions.append(actions)
                batch_masks.append(masks)
            stacked_masks = torch.stack(batch_masks)
            if stacked_masks.all():
                action = torch.zeros(n, self.A, device=self.device).uniform_(self.a_low, self.a_high)
            else:
                action = self.theta(None, torch.stack(batch_states), torch.stack(batch_actions), stacked_masks)
        else:
            action = self.theta(curr_state)

        if explore:
            noise = torch.normal(0, self.exp_noise, action.shape).to(device=self.device)
            action += noise
            action = torch.clamp(action, self.a_low, self.a_high)

        return action.numpy(force=True)

    def update(self, curr_iter):
        S, A, R, S_, d, idx = self.D.sample(self.B)

        with torch.no_grad():
            if self.actor_is_transformer:
                hs, ha, hm = self.D.batch_get_history((idx + 1) % self.D_size)
                A_ = self.theta_target(None, hs, ha, hm)
            else:
                A_ = self.theta_target(S_)

            noise = torch.clamp(
                torch.normal(0, self.policy_noise, A_.shape, device=self.device),
                -self.noise_clip, self.noise_clip
            )
            A_ = torch.clamp(A_ + noise, self.a_low, self.a_high)
            target = R + self.gamma * (1 - d) * torch.min(
                self.phi1_target(S_, A_), self.phi2_target(S_, A_)
            )

        self.phi1_optimiser.zero_grad()
        loss1 = ((self.phi1(S, A) - target) ** 2).mean()
        loss1.backward()
        self.phi1_optimiser.step()

        self.phi2_optimiser.zero_grad()
        loss2 = ((self.phi2(S, A) - target) ** 2).mean()
        loss2.backward()
        self.phi2_optimiser.step()

        actor_loss = torch.tensor(0.0)
        if curr_iter % self.policy_delay == 0:
            self.actor_optimiser.zero_grad()
            if self.actor_is_transformer:
                hs, ha, hm = self.D.batch_get_history(idx)
                actor_loss = (-self.phi1(S, self.theta(None, hs, ha, hm))).mean()
            else:
                actor_loss = (-self.phi1(S, self.theta(S))).mean()
            actor_loss.backward()
            self.actor_optimiser.step()

            with torch.no_grad():
                for p, tp in zip(self.phi1.parameters(), self.phi1_target.parameters()):
                    tp.data.copy_(self.polyak_coeff * p.data + (1 - self.polyak_coeff) * tp.data)
                for p, tp in zip(self.phi2.parameters(), self.phi2_target.parameters()):
                    tp.data.copy_(self.polyak_coeff * p.data + (1 - self.polyak_coeff) * tp.data)
                for p, tp in zip(self.theta.parameters(), self.theta_target.parameters()):
                    tp.data.copy_(self.polyak_coeff * p.data + (1 - self.polyak_coeff) * tp.data)

        return loss1, loss2, actor_loss