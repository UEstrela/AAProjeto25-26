import random
from typing import Dict, Tuple
from src.core.interfaces import Agente
from src.core.definitions import Accao, Observacao, ModoOperacao

class AgenteAprendizagem(Agente):
    """[cite: 159]"""

    def __init__(self, id_agente: str):
        super().__init__(id_agente)
        self.tabela_q: Dict[Tuple, Dict[str, float]] = {} # Tabela Q (Estado -> {Acao -> Valor})
        
        # Hiperparâmetros de RL [cite: 216]
        self.alpha = 0.1   # Taxa de aprendizagem
        self.gamma = 0.9   # Fator de desconto
        self.epsilon = 1.0 # Taxa de exploração (decai com o tempo)
        
        # Estado interno
        self.estado_anterior = None
        self.accao_anterior = None

    @classmethod
    def cria(cls, nome_do_ficheiro: str) -> 'Agente':
        # Ler configurações do ficheiro e devolver instância
        return cls("Agente_RL_1")

    def observacao(self, obs: Observacao):
        """Guarda a observação atual (s_t)"""
        self.estado_atual = obs.dados # Simplificação: dados devem ser hashable (tuple/str)

    def age(self) -> Accao:
        """Estratégia Epsilon-Greedy [cite: 217]"""
        
        # Inicializar estado na tabela se não existir
        if self.estado_atual not in self.tabela_q:
            self.tabela_q[self.estado_atual] = {"MoverNorte": 0.0, "MoverSul": 0.0, "MoverEste": 0.0, "MoverOeste": 0.0}

        # Escolha da Ação
        if self.modo == ModoOperacao.APRENDIZAGEM and random.random() < self.epsilon:
            # Exploração (Aleatório) [cite: 217]
            nome_accao = random.choice(list(self.tabela_q[self.estado_atual].keys()))
        else:
            # Aproveitamento (Melhor Q) [cite: 217]
            qs = self.tabela_q[self.estado_atual]
            nome_accao = max(qs, key=qs.get)

        accao = Accao(nome_accao)
        
        # Guardar contexto para o passo de atualização (s, a)
        self.estado_anterior = self.estado_atual
        self.accao_anterior = nome_accao
        
        return accao

    def avaliacaoEstadoAtual(self, recompensa_extrinseca: float):
        """
        Atualiza Tabela Q segundo Equation de Bellman [cite: 216]
        Inclui Recompensa Intrínseca (Novidade/Tempo) [cite: 218-234]
        """
        if self.modo != ModoOperacao.APRENDIZAGEM:
            return

        if self.estado_anterior is None or self.accao_anterior is None:
            return

        # 1. Calcular Recompensa Total
        # Penalização temporal constante [cite: 231]
        recompensa_tempo = -0.1 
        # (Futuro: Calcular novidade aqui para [cite: 234])
        recompensa_total = recompensa_extrinseca + recompensa_tempo

        # 2. Obter Max Q do próximo estado (s')
        if self.estado_atual not in self.tabela_q:
            self.tabela_q[self.estado_atual] = {"MoverNorte": 0.0, "MoverSul": 0.0, "MoverEste": 0.0, "MoverOeste": 0.0}
        
        max_q_proximo = max(self.tabela_q[self.estado_atual].values())

        # 3. Atualizar Q(s,a) - Fórmula Bellman [cite: 216]
        q_atual = self.tabela_q[self.estado_anterior][self.accao_anterior]
        
        novo_q = q_atual + self.alpha * (recompensa_total + (self.gamma * max_q_proximo) - q_atual)
        
        self.tabela_q[self.estado_anterior][self.accao_anterior] = novo_q

        # Decaimento do Epsilon (opcional, mas recomendado)
        if self.epsilon > 0.01:
            self.epsilon *= 0.995

    def comunica(self, mensagem: str, de_agente: Agente):
        """[cite: 180-182]"""
        print(f"[{self.id}] Recebi mensagem de {de_agente.id}: {mensagem}")