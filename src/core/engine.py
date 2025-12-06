import time
from typing import List
from src.core.interfaces import Agente, Ambiente
from src.core.definitions import ModoOperacao

class MotorDeSimulacao:
    """[cite: 132]"""
    
    def __init__(self, ambiente: Ambiente):
        self.agentes: List[Agente] = [] # [cite: 129]
        self.ambiente: Ambiente = ambiente # [cite: 130]
        self.passoAtual: int = 0 # [cite: 131]
        self.aExecutar: bool = False

    def adicionaAgente(self, agente: Agente):
        self.agentes.append(agente)

    def listaAgentes(self) -> List[Agente]:
        """[cite: 134]"""
        return self.agentes

    def executa(self, max_passos: int = 1000):
        """[cite: 135] - Implementa o ciclo do diagrama de sequência"""
        print(f"--- A iniciar Simulação (Motor) ---")
        self.aExecutar = True
        self.passoAtual = 0

        while self.aExecutar and self.passoAtual < max_passos:
            # 1. Atualizar Ambiente [cite: 196]
            self.ambiente.atualizacao()

            accoes_do_passo = {}

            # 2. Perceção e Deliberação [cite: 197]
            for agente in self.agentes:
                # Solicitar Estado Local / Devolver Percepção [cite: 206, 207]
                obs = self.ambiente.observacaoPara(agente)
                agente.observacao(obs)
                
                # Deliberar / Selecionar Ação [cite: 209]
                accao = agente.age()
                accoes_do_passo[agente] = accao

            # 3. Execução da Ação [cite: 199]
            for agente, accao in accoes_do_passo.items():
                # Tentar Executar / Devolver Resultado [cite: 210, 211]
                recompensa = self.ambiente.agir(accao, agente)
                
                # Agente avalia (Momento de Aprendizagem)
                agente.avaliacaoEstadoAtual(recompensa)

            # 4. Registar Estado / Métricas [cite: 200]
            # self.registarResultados()

            self.passoAtual += 1
            # time.sleep(0.1) # Opcional para visualização

        print("--- Terminar Simulação [cite: 202] ---")