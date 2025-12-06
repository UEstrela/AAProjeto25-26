import time
from typing import List
from src.core.interfaces import Agent, Environment
from src.core.definitions import OperationMode

class SimulationEngine:
    """[cite: 132]"""

    def __init__(self, environment: Environment):
        self.agents: List[Agent] = []  # [cite: 129]
        self.environment: Environment = environment  # [cite: 130]
        self.current_step: int = 0  # [cite: 131]
        self.is_running: bool = False

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def list_agents(self) -> List[Agent]:
        """[cite: 134]"""
        return self.agents

    def run(self, max_steps: int = 1000):
        """[cite: 135] - Implementa o ciclo do diagrama de sequência"""
        print("--- A iniciar Simulação (Engine) ---")
        self.is_running = True
        self.current_step = 0

        while self.is_running and self.current_step < max_steps:
            # 1. Atualizar Ambiente [cite: 196]
            self.environment.update()

            actions_this_step = {}

            # 2. Perceção e Deliberação [cite: 197]
            for agent in self.agents:
                # Solicitar Estado Local / Devolver Percepção [cite: 206, 207]
                obs = self.environment.observe_for(agent)
                agent.observe(obs)

                # Deliberar / Selecionar Ação [cite: 209]
                action = agent.act()
                actions_this_step[agent] = action

            # 3. Execução da Ação [cite: 199]
            for agent, action in actions_this_step.items():
                # Tentar Executar / Devolver Resultado [cite: 210, 211]
                reward = self.environment.act(action, agent)

                # Agente avalia (Momento de Aprendizagem)
                agent.evaluate_current_state(reward)

            # 4. Registar Estado / Métricas [cite: 200]
            # self.record_results()

            self.current_step += 1
            # time.sleep(0.1)  # Opcional para visualização

        print("--- Terminar Simulação [cite: 202] ---")
