from simulation import HumanoidSimulation


ENV_NAME = 'Humanoid-v5'

if __name__ == "__main__":
    simulation = HumanoidSimulation()
   # simulation.train()
    simulation.visualize(path_to_model="models/SAC_150000", iterations=10)








