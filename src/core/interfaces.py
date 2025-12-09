from abc import ABC, abstractmethod
from typing import Tuple, Any
from src.core.definitions import OperationMode

class Agent(ABC):
    mode: OperationMode

    @abstractmethod
    def observe(self, state: Tuple[int, int]):
        """Receive current state (x, y)"""
        pass

    @abstractmethod
    def act(self) -> str:
        """Return action as string (e.g. 'NORTH')"""
        pass

    @abstractmethod
    def evaluate_current_state(self, reward: float, next_state: Tuple[int, int]):
        """Update Q based on reward and next state"""
        pass

class Environment(ABC):
    @abstractmethod
    def get_state(self, agent: Agent) -> Tuple[int, int]:
        """Return current state (x, y)"""
        pass

    @abstractmethod
    def act(self, action: str, agent: Agent) -> float:
        """Execute action, update state, return reward"""
        pass

    @abstractmethod
    def reset(self):
        """Reset environment to initial state"""
        pass

    @abstractmethod
    def display(self):
        """Print environment to console"""
        pass
