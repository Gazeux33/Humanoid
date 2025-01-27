import gymnasium as gym
from stable_baselines3 import SAC
import os

from abc import ABC, abstractmethod


class HumanoidSimulationBase(ABC):
    def __init__(self,simulation_name:str ,env_name: str = 'Humanoid-v5', model_dir: str = "checkpoints"):
        self.simulation_name = simulation_name
        self.model = None
        self.env_name = env_name
        self.env = None
        self.agent = None
        self.model_dir = model_dir



    @abstractmethod
    def reward(self, state, action, next_state, rewards=None):
        raise NotImplementedError

    def train(self, checkpoints: str = None, save_freq = 50):
        self._init_env(self.env_name, mode="rgb_array")

        if checkpoints is not None:
            self.model = SAC.load(checkpoints, env=self.env)
            print(f"Loaded model from {checkpoints}")
        else:
            self.model = SAC('MlpPolicy', self.env, verbose=1)


        iters = 0
        while True:
            iters += 1

            obs, _ = self.env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                reward = self.reward(obs, action, next_obs, rewards=reward)
                self.model.learn(total_timesteps=1, reset_num_timesteps=False)
                obs = next_obs

            if iters % save_freq == 0:
                name = f"{self.model_dir}/{self.simulation_name}/SAC_{self.model.num_timesteps}"
                self.model.save(name)
                print(f"Saved model at {name}")

    def visualize(self, path_to_model: str, iterations: int = 100):
        self._init_env(self.env_name, mode="human")

        self.model = SAC.load(path_to_model, env=self.env)
        print(f"Loaded model from {path_to_model}")

        for _ in range(iterations):
            state, _ = self.env.reset()
            episode_over = False
            while not episode_over:
                action, _ = self.model.predict(state, deterministic=True)
                state, reward, terminated, truncated, info = self.env.step(action)
                episode_over = terminated or truncated
                self.env.render()
        self.env.close()


    def _init_env(self, name: str , mode: str):
        self.env = gym.make(name, render_mode=mode)
        observation, info = self.env.reset()
        print("Observation space:", self.env.observation_space)
        print("Action space:", self.env.action_space)
        return observation, info

