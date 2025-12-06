from abc import ABC, abstractmethod
from typing import Any
from src.core.definitions import Action, Observation, OperationMode

class Agent(ABC):
    """Classe abstrata para agentes."""
    mode: OperationMode

    @abstractmethod
    def create(self) -> 'Agent':
        """Factory method exigido: create(file_name_params)"""
        pass

    @abstractmethod
    def observe(self, obs: Observation):
        """Receive observation (observacao)"""
        pass

    @abstractmethod
    def act(self) -> Action:
        """Decide action (age)"""
        pass

    @abstractmethod
    def evaluate_current_state(self, reward: float):
        """Evaluate current state (avaliacaoEstadoAtual)"""
        pass

    @abstractmethod
    def communicate(self):
        """Communication method (comunica)"""
        pass

class Environment(ABC):
    """Classe abstrata para ambientes."""

    @abstractmethod
    def observe_for(self, agent: Agent) -> Observation:
        """Retrieve observation for agent (observacaoPara)"""
        pass

    @abstractmethod
    def act(self, action: Action, agent: Agent) -> float:
        """Execute action and return reward (agir)"""
        pass

    @abstractmethod
    def update(self):
        """Update environment (atualizacao)"""
        pass
