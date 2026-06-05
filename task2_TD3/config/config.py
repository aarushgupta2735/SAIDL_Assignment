from dataclasses import dataclass

@dataclass
class TD3config():

    ##env feartures: hardcoded for HopperV5
    obs_features: int = 11
    act_features: int = 3
    a_low: int = -1
    a_high: int = 1

    #switch
    use_transformer: bool = False

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

