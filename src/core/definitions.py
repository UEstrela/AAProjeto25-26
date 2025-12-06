from enum import Enum
from typing import Any

class OperationMode(Enum):
    """Definição do modo de operação [cite: 126]"""
    LEARNING = 1
    TEST = 2

class Action:
    """Estrutura de Ação [cite: 155]"""
    def __init__(self, name: str, parameters: Any = None):
        self.name = name
        self.parameters = parameters

    def __repr__(self):
        return f"Action({self.name})"

class Observation:
    """Estrutura de Observação [cite: 157]"""
    def __init__(self, data: Any):
        self.data = data
        self.timestamp = 0  # [cite: 158]

    def __repr__(self):
        return f"Observation({self.data})"
