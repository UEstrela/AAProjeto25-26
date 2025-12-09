import random
from typing import Dict, Tuple
from src.core.interfaces import Agent
from src.core.definitions import OperationMode

class LearningAgent(Agent):
    def __init__(self, agent_id: str):
        self.id = agent_id
        self.current_state = None
        self.previous_state = None
        self.previous_action = None
        self.q_table: Dict[Tuple[int, int], Dict[str, float]] = {}

        # Hyperparameters
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0

    def observe(self, state: Tuple[int, int]):
        """Store current state (x, y)"""
        self.current_state = state

        # Initialize Q for new states
        if state not in self.q_table:
            self.q_table[state] = {
                'NORTH': 0.0,
                'SOUTH': 0.0,
                'EAST': 0.0,
                'WEST': 0.0
            }

    def act(self) -> str:
        """Epsilon-greedy action selection"""
        state = self.current_state

        # Epsilon-greedy
        if self.mode == OperationMode.LEARNING and random.random() < self.epsilon:
            # Explore: random action
            action = random.choice(['NORTH', 'SOUTH', 'EAST', 'WEST'])
        else:
            # Exploit: best action
            action = max(self.q_table[state], key=self.q_table[state].get)

        # Store for learning
        self.previous_state = state
        self.previous_action = action

        return action

    def evaluate_current_state(self, reward: float, next_state: Tuple[int, int]):
        """Q-learning update: Q(s,a) += alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))"""
        if self.mode != OperationMode.LEARNING:
            return

        s = self.previous_state
        a = self.previous_action

        if s is None or a is None:
            return

        # Ensure next_state in Q
        if next_state not in self.q_table:
            self.q_table[next_state] = {
                'NORTH': 0.0,
                'SOUTH': 0.0,
                'EAST': 0.0,
                'WEST': 0.0
            }

        # Compute max Q for next state
        max_q_next = max(self.q_table[next_state].values())

        # Q-learning formula
        td_error = reward + self.gamma * max_q_next - self.q_table[s][a]
        self.q_table[s][a] += self.alpha * td_error

        # Decay epsilon
        if self.epsilon > 0.01:
            self.epsilon *= 0.996  # Slight decay
