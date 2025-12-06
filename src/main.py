import time
from src.core.engine import MotorDeSimulacao
from src.environments.lighthouse.py import AmbienteFarol
from src.agents.learning_agent import AgenteAprendizagem
from src.core.definitions import ModoOperacao

def main():
    # 1. Configuração Inicial [cite: 322]
    env = AmbienteFarol()
    motor = MotorDeSimulacao(ambiente=env)

    # 2. Criar Agente
    agente = AgenteAprendizagem("Agente_01")
    agente.modo = ModoOperacao.APRENDIZAGEM # Ativar modo de treino [cite: 346]
    motor.adicionaAgente(agente)

    # 3. Executar Simulação (Treino - Vários Episódios)
    n_episodios = 50
    print(f"--- A iniciar Treino: {n_episodios} Episódios ---")

    for ep in range(n_episodios):
        # Resetar posição do agente para (0,0) manualmente para cada episódio
        env.posicoes_agentes[agente] = (0, 0) 
        
        # Correr o motor até o agente chegar ao alvo ou estourar o limite de passos
        motor.executa(max_passos=100)
        
        # Verificar se chegou (para estatística)
        pos_final = env.posicoes_agentes[agente]
        if pos_final == env.pos_farol:
            print(f"Episódio {ep+1}: SUCESSO! (Epsilon: {agente.epsilon:.3f})")
        else:
            print(f"Episódio {ep+1}: Falhou.")

    # 4. Modo de Teste (Ver o resultado final) [cite: 370]
    print("\n--- A iniciar Teste Final (Visual) ---")
    agente.modo = ModoOperacao.TESTE # Desliga exploração
    env.posicoes_agentes[agente] = (0, 0)
    
    # Redefinir executa para ser mais lento e visível
    # Nota: Podes precisar de ajustar o engine.py para aceitar 'delay' ou fazer sleep aqui
    motor.executa(max_passos=20) 
    env.display() # Mostrar estado final

if __name__ == "__main__":
    main()