from typing import Tuple
from src.core.interfaces import Environment, Agent

class LighthouseEnvironment(Environment):
    def __init__(self):
        self.grid_size = 10
        self.lighthouse_pos = (9, 9)
        self.agent_pos = (0, 0)  # Single agent

    def get_state(self, agent: Agent) -> Tuple[int, int]:
        """Return current agent position"""
        return self.agent_pos

    def act(self, action: str, agent: Agent) -> float:
        """Execute action, update position, return reward"""
        x, y = self.agent_pos

        if action == 'MoveNorth':
            y = max(0, y - 1)
        elif action == 'MoveSouth':
            y = min(self.grid_size - 1, y + 1)
        elif action == 'MoveEast':
            x = min(self.grid_size - 1, x + 1)
        elif action == 'MoveWest':
            x = max(0, x - 1)

        self.agent_pos = (x, y)

        # Reward
        if self.agent_pos == self.lighthouse_pos:
            return 100.0  # Goal reached
        else:
            return -0.1   # Step penalty

    def reset(self):
        """Reset agent to start"""
        self.agent_pos = (0, 0)

    def is_at_goal(self) -> bool:
        """Check if agent reached lighthouse"""
        return self.agent_pos == self.lighthouse_pos

    def display(self):
        """Print simple grid"""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.lighthouse_pos[1]][self.lighthouse_pos[0]] = 'L'  # F for farol
        grid[self.agent_pos[1]][self.agent_pos[0]] = 'A'

        print("\n" + "-" * (self.grid_size * 2 + 2))
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("-" * (self.grid_size * 2 + 2))
