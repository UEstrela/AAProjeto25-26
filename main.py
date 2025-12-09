from src.core.engine import SimulationEngine
from src.environments.lighthouse import LighthouseEnvironment
from src.agents.learning_agent import LearningAgent
from src.core.definitions import OperationMode

def main():
    # Initialize
    environment = LighthouseEnvironment()
    engine = SimulationEngine(environment)

    agent = LearningAgent("Agent_01")
    agent.mode = OperationMode.LEARNING
    engine.add_agent(agent)

    # Train
    print("Starting Training...")
    engine.run_training(num_episodes=100, max_steps_per_episode=100)

    # Test
    print("\nStarting Test...")
    agent.mode = OperationMode.TEST
    engine.run_test()

if __name__ == "__main__":
    main()
