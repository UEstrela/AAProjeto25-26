from enum import Enum
from typing import Any

class ModoOperacao(Enum):
    """Definido no diagrama UML [cite: 126]"""
    APRENDIZAGEM = 1
    TESTE = 2

class Accao:
    """Estrutura de Ação [cite: 155]"""
    def __init__(self, nome: str, parametros: Any = None):
        self.nome = nome
        self.parametros = parametros

    def __repr__(self):
        return f"Accao({self.nome})"

class Observacao:
    """Estrutura de Observação [cite: 157]"""
    def __init__(self, dados: Any):
        self.dados = dados
        self.timestamp = 0 # [cite: 158]

    def __repr__(self):
        return f"Observacao({self.dados})"