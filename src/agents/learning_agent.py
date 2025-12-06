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

        # Lazy initialization: if state s not in q_table, init with 0.0 for all actions
        if self.current_state not in self.q_table:
            self.q_table[self.current_state] = {"MoveNorth": 0.0, "MoveSouth": 0.0, "MoveEast": 0.0, "MoveWest": 0.0}

        # Action selection: exploration if random < epsilon, else exploitation
        temp_epsilon = self.epsilon if self.mode == OperationMode.LEARNING else 0.0
        if random.random() < temp_epsilon:
            # Exploration (purely random among 4 possible actions)
            action_name = random.choice(["MoveNorth", "MoveSouth", "MoveEast", "MoveWest"])
        else:
            # Exploitation (choose action with max Q for current state)
            action_name = max(self.q_table[self.current_state], key=self.q_table[self.current_state].get)

        action = Action(action_name)

        # Save context (s, a) for later Q-update
        self.previous_state = self.current_state
        self.previous_action = action_name

        return action

    def evaluate_current_state(self, extrinsic_reward: float):
        """
        Q-Learning algorithm update according to Bellman Equation.
        Q(s,a) <- Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
        """
        if self.mode != OperationMode.LEARNING:
            return

        if self.previous_state is None or self.previous_action is None:
            return

        # Calculate total reward (extrinsic + intrinsic penalty)
        time_penalty = -0.1
        total_reward = extrinsic_reward + time_penalty

        # Ensure s' (current_state) is initialized in Q-table if not visited
        if self.current_state not in self.q_table:
            self.q_table[self.current_state] = {"MoveNorth": 0.0, "MoveSouth": 0.0, "MoveEast": 0.0, "MoveWest": 0.0}

        # Get max over all possible actions for s'
        max_q_next = max(self.q_table[self.current_state].values())

        # Q-Learning update
        current_q = self.q_table[self.previous_state][self.previous_action]
        td_error = total_reward + self.gamma * max_q_next - current_q
        self.q_table[self.previous_state][self.previous_action] += self.alpha * td_error

        # Decay epsilon
        if self.epsilon > 0.01:
            self.epsilon *= 0.995

    def communicate(self, message: str, from_agent: Agent):
        """[cite: 180-182]"""
        print(f"[{self.id}] Received message from {from_agent.id}: {message}")
