import math
from typing import Dict, Tuple
from src.core.interfaces import Environment, Agent
from src.core.definitions import Action, Observation

class LighthouseEnvironment(Environment):
    """
    Ambiente de grelha simples (GridWorld) para o problema do Farol.
    Objetivo: Chegar à coordenada do Farol (9,9).
    """
    def __init__(self):
        self.size = 10
        self.lighthouse_pos = (9, 9)
        # Map agent -> (x, y)
        self.agent_positions: Dict[Agent, Tuple[int, int]] = {}

    def observe_for(self, agent: Agent) -> Observation:
        """
        Return current agent position.
        In the future, it could be a direction vector to the lighthouse.
        """
        if agent not in self.agent_positions:
            self.agent_positions[agent] = (0, 0)  # Start at (0,0)

        pos = self.agent_positions[agent]
        # Observation is the state (x, y) for Q-table
        return Observation(pos)

    def update(self):
        """The environment is static, the lighthouse does not move."""
        pass

    def act(self, action: Action, agent: Agent) -> float:
        """
        Execute movement and compute reward [cite: 351-364].
        """
        x, y = self.agent_positions[agent]

        # Movement logic
        if action.name == "MoveNorth": y = max(0, y - 1)
        elif action.name == "MoveSouth": y = min(self.size - 1, y + 1)
        elif action.name == "MoveEast": x = min(self.size - 1, x + 1)
        elif action.name == "MoveWest": x = max(0, x - 1)

        self.agent_positions[agent] = (x, y)

        # Reward computation
        # 1. Reached the lighthouse?
        if (x, y) == self.lighthouse_pos:
            return 100.0  # Extrinsic positive reward [cite: 356]

        # 2. Time penalty (step cost)
        return -0.1  # [cite: 363]

    def display(self):
        """Simple ASCII visualization for console"""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]

        # Draw lighthouse
        fx, fy = self.lighthouse_pos
        grid[fy][fx] = 'F'

        # Draw agents
        for agent, (ax, ay) in self.agent_positions.items():
            grid[ay][ax] = 'A'

        print("\n" + "-" * (self.size + 2))
        for row in grid:
            print("|" + "".join(row) + "|")
        print("-" * (self.size + 2))
