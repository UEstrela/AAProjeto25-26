import math
from typing import Dict, Tuple
from src.core.interfaces import Ambiente, Agente
from src.core.definitions import Accao, Observacao

class AmbienteFarol(Ambiente):
    """
    Ambiente de grelha simples (GridWorld) para o problema do Farol.
    Objetivo: Chegar à coordenada do Farol (9,9).
    """
    def __init__(self):
        self.tamanho = 10
        self.pos_farol = (9, 9)
        # Mapeia agente -> (x, y)
        self.posicoes_agentes: Dict[Agente, Tuple[int, int]] = {}
        
    def observacaoPara(self, agente: Agente) -> Observacao:
        """
        Retorna a posição atual do agente.
        No futuro, pode ser um vetor de direção para o farol.
        """
        if agente not in self.posicoes_agentes:
            self.posicoes_agentes[agente] = (0, 0) # Início em (0,0)
            
        pos = self.posicoes_agentes[agente]
        # Observação é o estado (x, y) para usar na Tabela Q
        return Observacao(pos)

    def atualizacao(self):
        """O ambiente é estático, o farol não se move."""
        pass

    def agir(self, accao: Accao, agente: Agente) -> float:
        """
        Executa movimento e calcula recompensa [cite: 351-364].
        """
        x, y = self.posicoes_agentes[agente]
        
        # Lógica de Movimento
        if accao.nome == "MoverNorte": y = max(0, y - 1)
        elif accao.nome == "MoverSul": y = min(self.tamanho - 1, y + 1)
        elif accao.nome == "MoverEste": x = min(self.tamanho - 1, x + 1)
        elif accao.nome == "MoverOeste": x = max(0, x - 1)
        
        self.posicoes_agentes[agente] = (x, y)
        
        # Cálculo da Recompensa
        # 1. Chegou ao Farol?
        if (x, y) == self.pos_farol:
            return 100.0 # Recompensa Extrínseca Positiva [cite: 356]
            
        # 2. Penalização por Tempo (Custo de passo)
        return -0.1 # [cite: 363]

    def display(self):
        """Visualização ASCII simples para a consola"""
        grid = [['.' for _ in range(self.tamanho)] for _ in range(self.tamanho)]
        
        # Desenhar Farol
        fx, fy = self.pos_farol
        grid[fy][fx] = 'F'
        
        # Desenhar Agentes
        for agente, (ax, ay) in self.posicoes_agentes.items():
            grid[ay][ax] = 'A'
            
        print("\n" + "-" * (self.tamanho + 2))
        for linha in grid:
            print("|" + "".join(linha) + "|")
        print("-" * (self.tamanho + 2))