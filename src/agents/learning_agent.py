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
                'MoveNorth': 0.0,
                'MoveSouth': 0.0,
                'MoveEast': 0.0,
                'MoveWest': 0.0
            }

    def act(self) -> str:
        """Seleção de ação epsilon-greedy"""
        state = self.current_state
        actions = ['MoveNorth', 'MoveSouth', 'MoveEast', 'MoveWest']

        # Epsilon-greedy
        if self.mode == OperationMode.LEARNING and random.random() < self.epsilon:
            # Exploração: ação aleatória
            action = random.choice(actions)
        else:
            # Aproveitamento: ação com maior Q
            action = max(self.q_table[state], key=self.q_table[state].get)

        # Guardar para aprendizagem posterior
        self.previous_state = state
        self.previous_action = action

        return action

    def learn(self, s, a, r, s_next):
        """Q-Learning update: Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s_next,:)) - Q(s,a))"""
        if s is None or a is None:
            return

        # Inicializar Q para novo estado s_next se necessário
        if s_next not in self.q_table:
            self.q_table[s_next] = {
                'MoveNorth': 0.0,
                'MoveSouth': 0.0,
                'MoveEast': 0.0,
                'MoveWest': 0.0
            }

        # Máximo Q para s_next
        max_q_next = max(self.q_table[s_next].values())

        # Fórmula Q-Learning
        td_target = r + self.gamma * max_q_next
        td_error = td_target - self.q_table[s][a]
        self.q_table[s][a] += self.alpha * td_error
