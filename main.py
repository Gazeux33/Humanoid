import glfw

from simulations import DefaultHumanoidSimulation

glfw.init()

ENV_NAME = 'Humanoid-v5'

if __name__ == "__main__":
    name = "DefaultHumanoid"
    simulation = DefaultHumanoidSimulation(name)
    # simulation.train()
    simulation.visualize(path_to_model=f"checkpoints/{name}/SAC_150", iterations=10)








