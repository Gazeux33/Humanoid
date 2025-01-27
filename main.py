from src.simulation import DefaultHumanoidSimulation
from src.utils import get_last_save_path






ENV_NAME = 'Humanoid-v5'

if __name__ == "__main__":
    name = "DefaultHumanoid"
    simulation = DefaultHumanoidSimulation(name)
    last_save_path = get_last_save_path('checkpoints/DefaultHumanoid')
    #simulation.train(checkpoints=last_save_path, save_freq=50)
    simulation.visualize(path_to_model=last_save_path, iterations=10)










