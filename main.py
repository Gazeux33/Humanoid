import argparse
from src.simulation import HumanoidSimulationBase
from src.utils import get_last_save_path

ENV_NAME = 'Humanoid-v5'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Humanoid Simulation")
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_parser = subparsers.add_parser('train', help='Train the humanoid simulation')
    train_parser.add_argument('--checkpoints', type=str, default='checkpoints/DefaultHumanoid', help='Path to checkpoints')
    train_parser.add_argument('--save_freq', type=int, default=50, help='Save frequency')
    train_parser.add_argument('--max_episodes', type=int, default=1000, help='Maximum number of episodes')

    visualize_parser = subparsers.add_parser('visualize', help='Visualize the humanoid simulation')
    visualize_parser.add_argument('--path_to_model', type=str, required=True, help='Path to the model')
    visualize_parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    name = "DefaultHumanoid"
    simulation = HumanoidSimulationBase(name)

    if args.command == 'train':
        last_save_path = get_last_save_path(args.checkpoints)
        simulation.train(checkpoints=last_save_path, save_freq=args.save_freq, max_episodes=args.max_episodes)
    elif args.command == 'visualize':
        simulation.visualize(path_to_model=args.path_to_model, iterations=args.iterations)