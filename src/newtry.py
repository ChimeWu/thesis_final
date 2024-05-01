from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import pandas as pd
from os.path import abspath, dirname, join
from env import RyeFlexEnv
import random
import numpy as np
from test_sbx_model import *
from stable_baselines3.ddpg.policies import MlpPolicy

class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomNetwork, self).__init__(observation_space, features_dim)
        self.lstm = nn.LSTM(observation_space.shape[0], features_dim)
        self.features_dim = features_dim

    def forward(self, observations):
        return self.lstm(observations)

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, features_extractor_class=CustomNetwork, features_extractor_kwargs=dict(features_dim=256))

# 使用自定义策略

# 使用自定义策略
def train_sbx_ddpg(timesteps):
    policy_kwargs = dict(
        features_extractor_class=CustomNetwork,
        features_extractor_kwargs=dict(features_dim=256),
    )

    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env,policy_kwargs=policy_kwargs, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save("sbx_ddpg_rye_flex_env")

def continue_train_sbx_ddpg(timesteps,env):
    model = DDPG.load("sbx_ddpg_rye_flex_env")
    model.set_env(env)
    model.learn(total_timesteps=timesteps)
    model.save("sbx_ddpg_rye_flex_env")


root_dir = dirname(abspath(join(__file__, "../")))
data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
env = RyeFlexEnv(data)
train_sbx_ddpg(6000)
possible_times = env.get_possible_start_times()
possible_times = random.sample(possible_times, 50)
rewards = test_ddpg(possible_times,env)
mean_reward = np.mean(rewards)
print(mean_reward)
td3_rewards = test_td3(possible_times,env)
mean_reward = np.mean(td3_rewards)
print(mean_reward)

