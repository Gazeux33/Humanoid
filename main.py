import os

from src.simulations import DefaultHumanoidSimulation



def get_last(path:str)-> str:
    return os.path.join(path,max(os.listdir(path), key=lambda x: int(x.split('_')[1].split('.')[0])).replace('.zip', ''))


ENV_NAME = 'Humanoid-v5'

if __name__ == "__main__":
    name = "DefaultHumanoid"
    simulation = DefaultHumanoidSimulation(name)
    last_save_path = get_last('checkpoints/DefaultHumanoid')
    #simulation.train(checkpoints=last_save_path, save_freq=50)
    simulation.visualize(path_to_model=last_save_path, iterations=10)










