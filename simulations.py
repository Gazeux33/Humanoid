from simulation_base import HumanoidSimulationBase



class DefaultHumanoidSimulation(HumanoidSimulationBase):
    def __init__(self, name):
        super().__init__(name)

    def reward(self, state, action, next_state, rewards=None):
        return rewards






