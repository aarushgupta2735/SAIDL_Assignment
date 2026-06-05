from dataclasses import dataclass

@dataclass
class TD3config():

    ##env feartures: hardcoded for HopperV5
    #obs_features as property
    act_features: int = 3
    a_low: int = -1
    a_high: int = 1

    #switch
    use_transformer: bool = False
    is_velocity_hidden: bool = False 
    add_observation_noise: bool = False
    delay_rewards: bool = False
    on_RLHF: bool = False
    include_x_vel: bool = False

    ##TD3 hyperparamters
    D_size: int = 10**6
    lr: float = 3*(10**-4)
    batch_size:int = 256
    polyak_coeff: float = 0.005
    policy_delay: int = 2
    gamma: float = 0.99
    policy_noise: float = 0.2 #std of target action noise
    noise_clip: float = 0.5 #clips target policy smoothing noise
    exploration_noise: float = 0.1 #std of noise added during exploration

    #train config
    training_iterations:int = 1_000_000
    n_envs:int = 3
    BASE_SEED:int = 42
    eval_iterations: int = 5000

    #partial observability
        #Apply each modification independently to Hopper-v5; compare MLP-TD3 vs. TransformerTD3 (best L and L=32).
    observation_noise_std: int = 0.1 #0.1,0.3

    @property
    def obs_features(self):
        x = 0
        if(self.include_x_vel):
            x = 1
        if(self.is_velocity_hidden):
            return 11-2+x
        return 11+x