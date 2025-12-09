from typing import List
from src.core.interfaces import Agent, Environment
from src.core.definitions import OperationMode

class SimulationEngine:
    def __init__(self, environment: Environment):
        self.environment: Environment = environment
        self.agents: List[Agent] = []

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def run_episode(self, max_steps: int = 100) -> bool:
        """Run one episode synchronously"""
        steps = 0
        while steps < max_steps:
            # 1. Observe current state
            for agent in self.agents:
                state = self.environment.get_state(agent)
                agent.observe(state)

            # 2. Agents decide actions
            actions = {}
            for agent in self.agents:
                action = agent.act()
                actions[agent] = action

            # 3. Environment executes actions and computes rewards
            rewards = {}
            total_reward = 0.0
            for agent in self.agents:
                reward = self.environment.act(actions[agent], agent)
                rewards[agent] = reward
                total_reward += reward

            # 4. Observe next state and learn
            for agent in self.agents:
                next_state = self.environment.get_state(agent)
                agent.evaluate_current_state(rewards[agent], next_state)

            steps += 1

            # Check if goal (assume environment has method)
            if self.environment.is_at_goal():
                return True

        return False  # Not reached

    def run_training(self, num_episodes: int, max_steps_per_episode: int = 100):
        """Train for multiple episodes"""
        for ep in range(num_episodes):
            self.environment.reset()
            success = self.run_episode(max_steps_per_episode)
            print(f"Episode {ep+1}: {'Success' if success else 'Failed'}")

    def run_test(self):
        """Run test episode"""
        self.environment.reset()
        self.run_episode(max_steps=50)  # Or until goal
        self.environment.display()
