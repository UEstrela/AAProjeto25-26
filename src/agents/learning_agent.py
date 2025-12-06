import random
from typing import Dict, Tuple
from src.core.interfaces import Agent
from src.core.definitions import Action, Observation, OperationMode

class LearningAgent(Agent):
    """[cite: 159]"""

    def __init__(self, agent_id: str):
        self.id = agent_id
        self.q_table: Dict[Tuple, Dict[str, float]] = {}  # Tabela Q (Estado -> {Action -> Value})

        # Hiperparâmetros de RL [cite: 216]
        self.alpha = 0.1   # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 1.0 # Exploration rate (decays over time)

        # Internal state
        self.previous_state = None
        self.previous_action = None

    @classmethod
    def create(cls, config_file: str) -> 'Agent':
        # Load configurations from file and return instance
        return cls("LearningAgent_1")

    def observe(self, obs: Observation):
        """Store current observation (s_t)"""
        self.current_state = obs.data  # Simplified: data should be hashable (tuple/str)

    def act(self) -> Action:
        """Epsilon-Greedy strategy [cite: 217]"""

        # Initialize state in Q-table if not present
        if self.current_state not in self.q_table:
            self.q_table[self.current_state] = {"MoveNorth": 0.0, "MoveSouth": 0.0, "MoveEast": 0.0, "MoveWest": 0.0}

        # Action selection
        temp_epsilon = self.epsilon if self.mode == OperationMode.LEARNING else 0.0
        if random.random() < temp_epsilon:
            # Exploration (Random) [cite: 217]
            action_name = random.choice(list(self.q_table[self.current_state].keys()))
        else:
            # Exploitation (Best Q) [cite: 217]
            qs = self.q_table[self.current_state]
            action_name = max(qs, key=qs.get)

        action = Action(action_name)

        # Save context for update step (s, a)
        self.previous_state = self.current_state
        self.previous_action = action_name

        return action

    def evaluate_current_state(self, extrinsic_reward: float):
        """
        Update Q-table according to Bellman Equation [cite: 216]
        Includes Intrinsic Reward (Novelty/Time) [cite: 218-234]
        """
        if self.mode != OperationMode.LEARNING:
            return

        if self.previous_state is None or self.previous_action is None:
            return

        # 1. Compute Total Reward
        # Constant temporal penalty [cite: 231]
        time_penalty = -0.1
        # (Future: Compute novelty here for [cite: 234])
        total_reward = extrinsic_reward + time_penalty

        # 2. Get Max Q for next state (s')
        if self.current_state not in self.q_table:
            self.q_table[self.current_state] = {"MoveNorth": 0.0, "MoveSouth": 0.0, "MoveEast": 0.0, "MoveWest": 0.0}

        max_q_next = max(self.q_table[self.current_state].values())

        # 3. Update Q(s,a) - Bellman Formula [cite: 216]
        current_q = self.q_table[self.previous_state][self.previous_action]

        new_q = current_q + self.alpha * (total_reward + (self.gamma * max_q_next) - current_q)

        self.q_table[self.previous_state][self.previous_action] = new_q

        # Epsilon decay (optional, but recommended)
        if self.epsilon > 0.01:
            self.epsilon *= 0.995

    def communicate(self, message: str, from_agent: Agent):
        """[cite: 180-182]"""
        print(f"[{self.id}] Received message from {from_agent.id}: {message}")
