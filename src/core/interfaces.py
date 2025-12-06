from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.core.actions import Action, Observation

class Sensor(ABC):
    """Interface para sensores que filtram a informação do ambiente."""
    @abstractmethod
    def read(self, environment, agent_position) -> Observation:
        pass

class Agent(ABC):
    """
    Classe base para todos os agentes (Reativos ou Aprendizagem).
    Cumpre os requisitos do enunciado: observação, decisão (age), avaliação.
    """
    def __init__(self, name: str):
        self.name = name
        self.sensors: List[Sensor] = []
        self.total_reward = 0.0

    @classmethod
    @abstractmethod
    def create(cls, config_file: str) -> 'Agent':
        """Factory method exigido: cria(nome_do_ficheiro_parametros)"""
        pass

    @abstractmethod
    def observe(self, observation: Observation):
        """Recebe a percepção (observacao)"""
        pass

    @abstractmethod
    def act(self) -> Action:
        """Decide a ação a tomar (age)"""
        pass

    @abstractmethod
    def update_state(self, reward: float, done: bool = False):
        """
        Recebe o feedback da ação anterior. 
        É aqui que o Q-Learning será implementado (avaliacaoEstadoAtual).
        """
        pass

    def install_sensor(self, sensor: Sensor):
        """Método instala(sensor)"""
        self.sensors.append(sensor)

class Environment(ABC):
    """
    Classe base para os ambientes (Farol, Labirinto).
    """
    def __init__(self):
        self.step_count = 0

    @abstractmethod
    def get_observation(self, agent: Agent) -> Observation:
        """Gera a observação específica para um agente (observacaoPara)"""
        pass

    @abstractmethod
    def perform_action(self, agent: Agent, action: Action) -> float:
        """
        Executa a ação e retorna a recompensa imediata (agir).
        """
        pass

    @abstractmethod
    def update(self):
        """Atualiza a dinâmica do mundo (ex: mover obstáculos)"""
        pass

    @abstractmethod
    def display(self):
        """Visualização do estado atual"""
        pass