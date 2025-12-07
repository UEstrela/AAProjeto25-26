import os
import time
from typing import Dict, Tuple
from src.core.interfaces import Environment, Agent, Action, Observation

class MazeEnvironment(Environment):
    def __init__(self):
        # 0=Path, 1=Wall, 2=Goal, 3=Start
        # Layout complexo para testar inteligencia
        self.grid = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 3, 0, 0, 1, 0, 0, 1],  # (1,1) is Start
            [1, 1, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1, 2, 1],  # (5,6) is Goal
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        
        # Maps Agent -> (row, col)
        self.agent_positions: Dict[Agent, Tuple[int, int]] = {}
        self.start_pos = (1, 1) # Matches '3' in grid

    def get_observation(self, agent: Agent) -> Observation:
        if agent not in self.agent_positions:
            self.agent_positions[agent] = self.start_pos
        
        # Return position as state (row, col)
        return Observation(self.agent_positions[agent])

    def execute_action(self, agent: Agent, action: Action) -> float:
        row, col = self.agent_positions[agent]
        new_r, new_c = row, col

        # Movement Logic (Grid coordinates: Row=Y, Col=X)
        if action.name == "MoveUp":    new_r -= 1
        elif action.name == "MoveDown":  new_r += 1
        elif action.name == "MoveLeft":  new_c -= 1
        elif action.name == "MoveRight": new_c += 1

        # 1. Check Boundaries & Walls
        # If invalid move, stay in place and get penalized
        if (new_r < 0 or new_r >= self.height or 
            new_c < 0 or new_c >= self.width or 
            self.grid[new_r][new_c] == 1): # 1 is Wall
            return -5.0 # Collision penalty

        # 2. Update Position
        self.agent_positions[agent] = (new_r, new_c)

        # 3. Check Goal
        cell_type = self.grid[new_r][new_c]
        if cell_type == 2: # Goal
            return 100.0 # Success reward
        
        # 4. Standard Step Cost (encourages shortest path)
        return -1.0

    def display(self):
        """Draws the maze in the console"""
        # Clear screen (optional, works on Windows/Mac/Linux)
        # os.system('cls' if os.name == 'nt' else 'clear') 
        
        output = "\n"
        for r in range(self.height):
            line = ""
            for c in range(self.width):
                # Check if agent is here
                agent_here = False
                for pos in self.agent_positions.values():
                    if pos == (r, c):
                        line += "🤖" # Agent Icon
                        agent_here = True
                        break
                
                if not agent_here:
                    val = self.grid[r][c]
                    if val == 1: line += "██" # Wall
                    elif val == 0: line += "  " # Path
                    elif val == 2: line += "🏁" # Goal
                    elif val == 3: line += "S " # Start
            output += "|" + line + "|\n"
        print(output)