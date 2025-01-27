import gymnasium as gym
from stable_baselines3 import SAC


class HumanoidSimulationBase:
    """
    Base class for humanoid simulations, handling training and visualization.
    """

    def __init__(self, simulation_name: str, env_name: str = 'Humanoid-v5', model_dir: str = "checkpoints"):
        """
        Initialize the simulation with a name, environment, and model directory.

        Args:
            simulation_name (str): Name of the simulation.
            env_name (str): Gym environment name. Defaults to 'Humanoid-v5'.
            model_dir (str): Directory to save/load models. Defaults to "checkpoints".
        """
        self.simulation_name = simulation_name
        self.model = None
        self.env_name = env_name
        self.env = None
        self.agent = None
        self.model_dir = model_dir

    def train(self, checkpoints: str = None, save_freq=50, max_episodes="inf"):
        """
        Train the humanoid simulation.

        Args:
            checkpoints (str, optional): Path to load checkpoints. Defaults to None.
            save_freq (int, optional): Frequency of saving models. Defaults to 50.
            max_episodes (int, optional): Maximum number of training episodes. Defaults to "inf".
        """
        self._init_env(self.env_name, mode="rgb_array")
        self._load_model(checkpoints)

        iters = 0
        while self.model.num_timesteps < max_episodes:
            iters += 1
            info = self._run_one_episode(learn=True, render=False, deterministic=False)
            if iters % save_freq == 0:
                self._save_model()
        self.env.close()

    def visualize(self, path_to_model: str, iterations: int = 100):
        """
        Visualize the humanoid simulation using a trained model.

        Args:
            path_to_model (str): Path to the trained model.
            iterations (int, optional): Number of visualization episodes. Defaults to 100.
        """
        self._init_env(self.env_name, mode="human")

        self.model = SAC.load(path_to_model, env=self.env)
        print(f"Loaded model from {path_to_model}")

        for _ in range(iterations):
            self._run_one_episode(learn=False, render=True, deterministic=True)
        self.env.close()

    def _run_one_episode(self, learn: bool, render: bool, deterministic: bool):
        """
        Execute a single episode of the simulation.

        Args:
            learn (bool): Whether to update the model during the episode.
            render (bool): Whether to render the environment.
            deterministic (bool): Whether to use deterministic actions.

        Returns:
            info: Information about the episode.
        """
        obs, _ = self.env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if learn:
                self.model.learn(total_timesteps=1, reset_num_timesteps=False)
            obs = next_obs
            if render:
                self.env.render()
        return info
    
    def _save_model(self):
        name = f"{self.model_dir}/{self.simulation_name}/SAC_{self.model.num_timesteps}"
        self.model.save(name)
        print(f"Saved model at {name}")
        


    def _load_model(self, path_to_model: str):
        """
        Load a model from a given path, or initialize a new model.

        Args:
            path_to_model (str): Path to the model. If None, initialize a new model
        """
        if path_to_model is not None:
            self.model = SAC.load(path_to_model, env=self.env)
            print(f"Loaded model from {path_to_model}")
        else:
            self.model = SAC('MlpPolicy', self.env, verbose=1)
        
    
    def _init_env(self, name: str , mode: str):
        """
        Initialize the environment with a given name and render mode.

        Args:
            name (str): Name of the environment.
            mode (str): Render mode.
        """
        self.env = gym.make(name, render_mode=mode)
        observation, info = self.env.reset()
        print("Observation space:", self.env.observation_space)
        print("Action space:", self.env.action_space)
        return observation, info

