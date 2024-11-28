import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C

from ppo import PPO


class HumanoidSimulation:
    def __init__(self, env_name: str = 'Humanoid-v5', model_dir: str = "models"):
        self.model = None
        self.env_name = env_name
        self.env = None
        self.agent = None
        self.model_dir = model_dir

    def train(self, timesteps=25000, max_iters=100):
        self._init_env(self.env_name, mode="rgb_array")
        self.model = SAC('MlpPolicy', self.env, verbose=1)
        self.agent = PPO(self.env)

        iters = 0
        while iters < max_iters:
            iters += 1
            self.model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
            self.model.save(f"{self.model_dir}/SAC_{timesteps * iters}")


    def visualize(self, path_to_model: str,iterations: int = 100, ):
        self.model = SAC.load(path_to_model, env=self.env)
        self._init_env(self.env_name, mode="human")
        self.agent = PPO(self.env)

        for i in range(iterations):
            self._run_one_episode()
        self.env.close()

    def _run_one_episode(self):
        state, _ = self.env.reset()
        episode_over = False
        while not episode_over:
            #action = self.agent.select_action(state)
            action, _ = self.model.predict(state)
            state, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()
            episode_over = terminated or truncated
        self.env.reset()

    def _init_env(self, name: str , mode: str):
        self.env = gym.make(name, render_mode=mode)
        observation, info = self.env.reset()
        print("Observation space:", self.env.observation_space)
        print("Action space:", self.env.action_space)
        return observation, info

