"""
Script principal de entrenamiento del agente de control de semáforos.
Entrena el agente usando Q-Learning y guarda resultados.
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import TrafficEnvironment
from agent import QLearningAgent
import time
import json
import os


def train_agent(
    num_episodes: int = 2000,
    max_steps_per_episode: int = 500,
    seed: int = 954,
    save_dir: str = "models",
    plot: bool = True
):
    """
    Entrena al agente de control de semáforos.
    """
    print("=" * 60)
    print("ENTRENAMIENTO DE AGENTE DE CONTROL DE SEMÁFOROS")
    print("=" * 60)
    print(f"\nConfiguración:")
    print(f"  - Episodios: {num_episodes}")
    print(f"  - Pasos por episodio: {max_steps_per_episode}")
    print(f"  - Semilla inicial: {seed}")
    print(f"  - Algoritmo: Q-Learning")
    print()
    
    # Crear entorno y agente
    env = TrafficEnvironment(max_vehicles=100, seed=seed)
    agent = QLearningAgent(
        state_space_size=env.get_state_space_size(),
        action_space_size=env.get_action_space_size(),
        learning_rate=0.3,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.996,
        epsilon_min=0.01
    )
    
    # Métricas de entrenamiento
    episode_rewards = []
    episode_avg_queue = []
    episode_lengths = []
    moving_avg_rewards = []
    
    start_time = time.time()
    
    # Bucle principal de entrenamiento
    for episode in range(num_episodes):
        # Cambiar semilla cada 100 episodios para variedad
        if episode % 100 == 0 and episode > 0:
            env.seed = seed + episode
        
        state = env.reset()
        state_index = env.discretize_state(state)
        
        episode_reward = 0
        total_vehicles = []
        
        for step in range(max_steps_per_episode):
            # Seleccionar acción
            action = agent.select_action(state_index, training=True)
            
            # Ejecutar acción
            next_state, reward, done, info = env.step(action)
            next_state_index = env.discretize_state(next_state)
            
            # Actualizar agente
            agent.update(state_index, action, reward, next_state_index, done)
            
            # Acumular métricas
            episode_reward += reward
            total_vehicles.append(info["total_vehicles"])
            
            # Transición
            state_index = next_state_index
            
            if done:
                break
        
        # Finalizar episodio
        agent.finish_episode()
        
        # Guardar métricas
        episode_rewards.append(episode_reward)
        episode_avg_queue.append(np.mean(total_vehicles))
        episode_lengths.append(step + 1)
        
        # Promedio móvil
        if len(episode_rewards) >= 50:
            moving_avg = np.mean(episode_rewards[-50:])
        else:
            moving_avg = np.mean(episode_rewards)
        moving_avg_rewards.append(moving_avg)
        
        # Imprimir progreso
        if (episode + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            avg_reward = np.mean(episode_rewards[-100:])
            avg_queue = np.mean(episode_avg_queue[-100:])
            
            print(f"Episodio {episode + 1}/{num_episodes}")
            print(f"  Recompensa promedio por episodio: {avg_reward:.2f}")
            print(f"  Cola promedio: {avg_queue:.2f} vehículos")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Tiempo transcurrido: {elapsed_time:.1f}s")
            print()
    
    training_time = time.time() - start_time
    print(f"\n{"=" * 60}")
    print(f"ENTRENAMIENTO COMPLETADO")
    print(f"{"=" * 60}")
    print(f"Tiempo total: {training_time:.1f} segundos")
    print(f"Tiempo por episodio: {training_time/num_episodes:.3f}s")
    print()
    
    # Guardar modelo
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "qlearning_traffic_agent.pkl")
    agent.save_model(model_path)
    
    # Guardar métricas
    metrics = {
        "episode_rewards": episode_rewards,
        "episode_avg_queue": episode_avg_queue,
        "episode_lengths": episode_lengths,
        "moving_avg_rewards": moving_avg_rewards,
        "training_time": training_time,
        "num_episodes": num_episodes,
        "config": {
            "learning_rate": agent.alpha,
            "discount_factor": agent.gamma,
            "epsilon_decay": agent.epsilon_decay,
            "max_vehicles": env.max_vehicles
        }
    }
    
    metrics_path = os.path.join(save_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        # Convertir arrays numpy a listas para JSON
        metrics_json = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metrics.items()
        }
        json.dump(metrics_json, f, indent=2)
    
    print(f"Métricas guardadas en: {metrics_path}")
    
    # Graficar resultados
    if plot:
        plot_training_results(
            episode_rewards,
            moving_avg_rewards,
            episode_avg_queue,
            agent.training_stats["epsilon_history"],
            save_dir
        )
    
    return agent, metrics


def plot_training_results(
    episode_rewards,
    moving_avg_rewards,
    episode_avg_queue,
    epsilon_history,
    save_dir
):
    """
    Genera gráficos de los resultados del entrenamiento.
    """
    # Recompensa por episodio
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.3, label="Recompensa por episodio")
    plt.plot(moving_avg_rewards, linewidth=2, label="Media móvil (50 episodios)")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa Total")
    plt.title("Evolución de la Recompensa", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_rewards.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Cola promedio por episodio
    plt.figure(figsize=(10, 6))
    plt.plot(episode_avg_queue, color="orange", alpha=0.6, label="Cola promedio")
    if len(episode_avg_queue) >= 50:
        moving_avg_queue = [np.mean(episode_avg_queue[max(0, i-49):i+1]) 
                        for i in range(len(episode_avg_queue))]
        plt.plot(moving_avg_queue, color="red", linewidth=2, 
                label="Media móvil (50 episodios)")
    plt.xlabel("Episodio")
    plt.ylabel("Vehículos Promedio en Cola")
    plt.title("Congestión Promedio", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_congestion.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Decaimiento de Epsilon
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_history, color="green")
    plt.xlabel("Episodio")
    plt.ylabel("Epsilon")
    plt.title("Exploración vs Explotación (Epsilon)", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_epsilon.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Mejora acumulada
    plt.figure(figsize=(10, 6))
    # Dividir en cuartiles
    quartile_size = len(episode_rewards) // 4
    quartile_avgs = []
    quartile_labels = []
    for i in range(4):
        start = i * quartile_size
        end = start + quartile_size if i < 3 else len(episode_rewards)
        quartile_avgs.append(np.mean(episode_rewards[start:end]))
        quartile_labels.append(f"Q{i+1}\n({start}-{end})")
    
    bars = plt.bar(range(4), quartile_avgs, color=["#ff9999", "#ffcc99", "#99ccff", "#99ff99"])
    plt.xlabel("Cuartil de Entrenamiento")
    plt.ylabel("Recompensa Promedio")
    plt.title("Mejora por Cuartil", fontsize=14, fontweight="bold")
    plt.xticks(range(4), quartile_labels)
    plt.grid(True, alpha=0.3, axis="y")
    
    # Añadir valores sobre las barras
    for i, (bar, val) in enumerate(zip(bars, quartile_avgs)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f"{val:.1f}",
                ha="center", va="bottom", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_quartiles.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Gráficos guardados en: {save_dir}/")
    print(f"  - training_rewards.png")
    print(f"  - training_congestion.png")
    print(f"  - training_epsilon.png")
    print(f"  - training_quartiles.png")


if __name__ == "__main__":
    # Configuración del entrenamiento
    CONFIG = {
        "num_episodes": 2000,  # Suficiente con espacio de 50 estados
        "max_steps_per_episode": 500,
        "seed": 954,
        "save_dir": "models",
        "plot": True
    }
    
    # Entrenar agente
    agent, metrics = train_agent(**CONFIG)
