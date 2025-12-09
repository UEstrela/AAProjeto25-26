from src.core.engine import SimulationEngine
from src.environments.lighthouse import LighthouseEnvironment
from src.agents.learning_agent import LearningAgent
from src.core.definitions import OperationMode

def main():
    # Inicializar ambiente e motor
    environment = LighthouseEnvironment()
    engine = SimulationEngine(environment)

    agent = LearningAgent("Agent_01")
    engine.add_agent(agent)

    # Ciclo de Treino: 200 episódios com epsilon decrescendo para melhor aprendizado
    print("Iniciando Treino (Exploração)...")
    agent.mode = OperationMode.LEARNING
    for ep in range(200):
        environment.reset()
        success = engine.run_episode(max_steps=100)
        print(f"Episódio {ep+1}: {'Sucesso' if success else 'Falha'} (Epsilon: {agent.epsilon:.3f})")

        # Decaimento do Epsilon para reduzir exploração gradualmente
        if agent.epsilon > 0.01:
            agent.epsilon *= 0.99

    # Episódio de Teste: modo TESTE, sem exploração, mostrando movimento passo a passo
    print("\nIniciando Teste (Aproveitamento)...")
    agent.mode = OperationMode.TEST
    environment.reset()

    print("Caminho aprendido (A = Agente, L = Farol):")
    steps = 0
    while steps < 100:
        # Observe state
        state = environment.get_state(agent)
        agent.observe(state)

        # Act (no learning)
        action = agent.act()

        # Execute action
        reward = environment.act(action, agent)

        # Display after move
        print(f"\nPasso {steps+1}: Ação '{action}' -> Recompensa {reward}")
        environment.display()

        steps += 1
        if environment.is_at_goal():
            print("Chegou ao farol!")
            break

        # Pequeno delay para visualização
        import time
        time.sleep(0.5)

    if not environment.is_at_goal():
        print("Não chegou ao farol em 100 passos.")

if __name__ == "__main__":
    main()
