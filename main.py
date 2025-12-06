import time
from src.core.engine import SimulationEngine
from src.environments.lighthouse import LighthouseEnvironment
from src.agents.learning_agent import LearningAgent
from src.core.definitions import OperationMode

def main():
    # 1. Initial Configuration [cite: 322]
    env = LighthouseEnvironment()
    engine = SimulationEngine(environment=env)

    # 2. Create Agent
    agent = LearningAgent("Agent_01")
    agent.mode = OperationMode.LEARNING  # Activate learning mode [cite: 346]
    engine.add_agent(agent)

    # 3. Run Simulation (Training - Multiple Episodes)
    num_episodes = 100
    print(f"--- Starting Training: {num_episodes} Episodes ---")

    for ep in range(num_episodes):
        # Reset agent position to (0,0) manually for each episode
        env.agent_positions[agent] = (0, 0)

        # Run engine until agent reaches goal or max steps
        engine.run(max_steps=150)

        # Check if reached (for stats)
        final_pos = env.agent_positions[agent]
        if final_pos == env.lighthouse_pos:
            print(f"Episode {ep+1}: SUCCESS! (Epsilon: {agent.epsilon:.3f})")
        else:
            print(f"Episode {ep+1}: Failed.")

    # 4. Test Mode (Show final result) [cite: 370]
    print("\n--- Starting Final Test (Visual) ---")
    agent.mode = OperationMode.TEST  # Disable exploration
    env.agent_positions[agent] = (0, 0)

    # Adjust run to be slower and visible
    # Note: May need to adjust engine.py to accept 'delay' or add sleep here
    engine.run(max_steps=20)
    env.display()  # Show final state

if __name__ == "__main__":
    main()
