import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from config.config import TD3config
from .Actor import Actor
from .Critic import Critic
from .ReplayBuffer import ReplayBuffer

class TD3(nn.Module):
    def __init__(self, config:TD3config,buffer,device):
        super().__init__()
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

        self.theta = Actor(config) #takes in states and gives actions : policy
        self.theta_target = copy.deepcopy(self.theta) 
        for param in self.theta_target.parameters():
            param.requires_grad = False

        self.phi1 = Critic(config)#given (s,a) return an integer
        self.phi1_target = copy.deepcopy(self.phi1)
        for param in self.phi1_target.parameters():
            param.requires_grad = False
            
        self.phi2 = Critic(config)
        self.phi2_target = copy.deepcopy(self.phi2)
        for param in self.phi2_target.parameters():
            param.requires_grad = False

        self.D = buffer#<s,a,r,s',d> 
        
        self.phi1_optimiser = torch.optim.Adam(self.phi1.parameters(), lr=config.lr)
        self.phi2_optimiser = torch.optim.Adam(self.phi2.parameters(), lr=config.lr)
        self.actor_optimiser = torch.optim.Adam(self.theta.parameters(), lr=config.lr)

        self.polyak_coeff = config.polyak_coeff
        self.policy_delay = config.policy_delay

    def select_action(self,curr_state,explore=True):
        action = self.theta(curr_state)

        ##add guassian noise
        if(explore):
            noise = torch.normal(0,self.exp_noise,action.shape)
            action+=noise
            action = torch.clip(action,self.a_low,self.a_high) 

        return torch.tensor.numpy(action)
    
    
    def update(self,curr_iter):
        S,A,R,S_,d = self.D.sample(self.B)
        #Find A_
        A_ = self.theta_target(S_)

        ##add guassian noise and clip it
        noise = torch.normal(0,self.policy_noise,A_.shape)
        noise = torch.clip(noise,-self.noise_clip,self.noise_clip)
        A_+=noise
        A_=torch.clip(A_,self.a_low,self.a_high)

        #compute losses and objective funtion for policy
        self.phi1_optimiser.zero_grad()
        self.phi2_optimiser.zero_grad()
        target = R + self.gamma*(1-d)*torch.min(self.phi1_target(torch.cat([S_,A_],dim=1)),self.phi2_target(torch.cat([S_,A_],dim=1))).detach()    
        loss1=((self.phi1(torch.cat([S,A],dim=1))-target)**2).mean()
        loss2=((self.phi2(torch.cat([S,A],dim=1))-target)**2).mean()

        #gradient descent step
        loss1.backward()
        loss2.backward()
        self.phi1_optimiser.step()
        self.phi2_optimiser.step()

        self.actor_optimiser.zero_grad()
        loss=(-self.phi1(torch.cat([S,self.theta(S)],dim=1))).mean()
        #at every policy_delay steps, update the policy and target networks
        if(curr_iter%self.policy_delay==0):
            
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