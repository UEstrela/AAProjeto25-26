import time
from typing import List
from src.core.interfaces import Agent, Environment

class SimulationEngine:
    """
    Motor que sincroniza a execução dos agentes e do ambiente.
    """
    def __init__(self, environment: Environment):
        self.environment = environment
        self.agents: List[Agent] = []
        self.running = False

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def run(self, max_steps: int = 1000, delay: float = 0.1):
        """
        Ciclo principal de simulação (executa).
        """
        print(f"--- A iniciar simulação com {len(self.agents)} agentes ---")
        self.running = True
        step = 0

        while self.running and step < max_steps:
            # 1. Atualizar dinâmica do ambiente (ex: recursos a crescer)
            self.environment.update()

            # 2. Ciclo de decisão dos agentes
            actions_to_perform = {}
            
            # Fase de Perceção e Decisão
            for agent in self.agents:
                # O ambiente gera a observação para este agente
                obs = self.environment.get_observation(agent)
                # O agente processa a observação
                agent.observe(obs)
                # O agente escolhe a ação
                action = agent.act()
                actions_to_perform[agent] = action

            # 3. Fase de Execução (Atuação)
            for agent, action in actions_to_perform.items():
                # O ambiente executa e devolve recompensa
                reward = self.environment.perform_action(agent, action)
                
                # O agente aprende com o resultado
                agent.update_state(reward)
                
                # Logging simples
                # print(f"Passo {step}: {agent.name} fez {action.name} -> R={reward}")

            # 4. Visualização
            self.environment.display()
            
            step += 1
            time.sleep(delay) # Controlo de velocidade para visualização

        print("--- Simulação terminada ---")