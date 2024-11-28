


class PPO:
    def __init__(self,env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space