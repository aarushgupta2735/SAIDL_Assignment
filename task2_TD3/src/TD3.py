import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from task1_decoder_only_transformer.config.transformer_config import TransformerConfig
from task2_TD3.config.config import TD3config
from task2_TD3.src.Actor import Actor
from task2_TD3.src.Critic import Critic
from task2_TD3.src.ReplayBuffer import ReplayBuffer



class TD3(nn.Module):
    def __init__(self, config:TD3config,tConfig:TransformerConfig,buffer,device):
        super().__init__()
        self.device = device
        self.O = config.obs_features
        self.A = config.act_features
        self.B = config.batch_size
        self.D_size = config.D_size
        self.gamma = config.gamma
        self.device = device
        self.exp_noise = config.exploration_noise
        self.policy_noise = config.policy_noise
        self.noise_clip = config.noise_clip

        self.a_low = config.a_low
        self.a_high = config.a_high
        
        self.on_RLHF = config.on_RLHF

        self.n_envs = config.n_envs
        self.theta = Actor(config, tConfig, device=device) #takes in states and gives actions : policy
        self.theta_target = copy.deepcopy(self.theta) 
        for param in self.theta_target.parameters():
            param.requires_grad = False

        self.phi1 = Critic(config,device=device)#given (s,a) return an integer
        self.phi1_target = copy.deepcopy(self.phi1)
        for param in self.phi1_target.parameters():
            param.requires_grad = False
            
        self.phi2 = Critic(config,device=device)
        self.phi2_target = copy.deepcopy(self.phi2)
        for param in self.phi2_target.parameters():
            param.requires_grad = False

        self.D = buffer#<s,a,r,s',d> 
        
        self.phi1_optimiser = torch.optim.Adam(self.phi1.parameters(), lr=config.lr)
        self.phi2_optimiser = torch.optim.Adam(self.phi2.parameters(), lr=config.lr)
        self.actor_optimiser = torch.optim.Adam(self.theta.parameters(), lr=config.lr)

        self.polyak_coeff = config.polyak_coeff
        self.policy_delay = config.policy_delay

    def select_action(self,curr_state,explore=True,n_envs_override=None):

        if(self.actor_is_transformer):
            #we need the history of curr_state
            batch_states = []
            batch_actions = []
            batch_masks = []
            n = self.n_envs
            if(n_envs_override!=None):
                n = 1
            for i in range(n):
                states,actions,masks = self.D.get_current_history(i)
                batch_states.append(states)
                batch_actions.append(actions)
                batch_masks.append(masks)
            action = self.theta(None,torch.stack(batch_states),torch.stack(batch_actions),torch.stack(batch_masks))
        else:
            action = self.theta(curr_state)
        ##add guassian noise
        if(explore):
            noise = torch.normal(0,self.exp_noise,action.shape).to(device=self.device)
            action+=noise
            action = torch.clip(action,self.a_low,self.a_high) 

        return action.numpy(force=True)

    def update(self,curr_iter):
        S,A,R,S_,d,idx = self.D.sample(self.B) #each of (self.B,) 

        batch_states,batch_actions,batch_masks = self.D.batch_get_history(idx+1)

        ##add guassian noise and clip it
        with torch.no_grad():
            #Find A_
            if(self.actor_is_transformer):
                A_ = self.theta_target(None,batch_states,batch_actions,batch_masks)
            else:
                A_ = self.theta_target(S_)
            noise = torch.normal(0,self.policy_noise,A_.shape).to(device=self.device)
            noise = torch.clip(noise,-self.noise_clip,self.noise_clip).to(device=self.device)
            A_+=noise
            A_=torch.clip(A_,self.a_low,self.a_high)

        #compute losses and objective funtion for policy
        self.phi1_optimiser.zero_grad()
        self.phi2_optimiser.zero_grad()
        with torch.no_grad():
            target = R + self.gamma*(1-d)*torch.min(self.phi1_target(S_,A_),self.phi2_target(S_,A_))
        loss1=((self.phi1(S,A,)-target)**2).mean()
        loss2=((self.phi2(S,A)-target)**2).mean()

        #gradient descent step
        loss1.backward()
        loss2.backward()
        self.phi1_optimiser.step()
        self.phi2_optimiser.step()
        loss = torch.tensor(0.0)
        #at every policy_delay steps, update the policy and target networks
        if(curr_iter%self.policy_delay==0):
            self.actor_optimiser.zero_grad()

            batch_states,batch_actions,batch_masks = self.D.batch_get_history(idx)
        
            if(self.actor_is_transformer):
                loss=(-self.phi1(S,self.theta(None,batch_states,batch_actions,batch_masks))).mean()
            else:
                loss=(-self.phi1(S,self.theta(S))).mean()

            loss.backward()
            self.actor_optimiser.step()
            #update target networks
            with torch.no_grad():
                for param, target_param in zip(self.phi1.parameters(), self.phi1_target.parameters()):
                    target_param.data.copy_(self.polyak_coeff * param.data + (1 - self.polyak_coeff) * target_param.data)

                for param, target_param in zip(self.phi2.parameters(), self.phi2_target.parameters()):
                    target_param.data.copy_(self.polyak_coeff * param.data + (1 - self.polyak_coeff) * target_param.data)

                for param, target_param in zip(self.theta.parameters(), self.theta_target.parameters()):
                    target_param.data.copy_(self.polyak_coeff * param.data + (1 - self.polyak_coeff) * target_param.data)

        return loss1,loss2,loss