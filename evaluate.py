"""
Script para evaluar el desempeño del agente entrenado.
Compara el agente entrenado vs una política de control fijo.
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import TrafficEnvironment
from agent import QLearningAgent
import json
import os


class FixedTimeController:
    """
    Controlador de semáforo con temporizador fijo.
    Cambia de fase cada N pasos sin considerar el tráfico.
    """
    
    def __init__(self, switch_interval: int = 20):
        self.switch_interval = switch_interval
        self.steps_since_switch = 0
    
    def select_action(self, state_index: int) -> int:
        """
        Selecciona acción basada en temporizador.
        """
        self.steps_since_switch += 1
        
        if self.steps_since_switch >= self.switch_interval:
            self.steps_since_switch = 0
            return 1  # Cambiar
        else:
            return 0  # Mantener
    
    def reset(self):
        """Reinicia el contador."""
        self.steps_since_switch = 0


def evaluate_agent(
    agent,
    num_episodes: int = 100,
    seeds: list = None,
    render: bool = False
):
    """
    Evalúa el desempeño del agente en múltiples episodios.
    """
    if seeds is None:
        seeds = range(1000, 1000 + num_episodes)
    
    env = TrafficEnvironment(max_vehicles=40)
    
    episode_rewards = []
    episode_avg_queues = []
    episode_max_queues = []
    episode_total_waiting = []
    action_counts = {"maintain": 0, "change": 0}
    
    for i, seed in enumerate(seeds[:num_episodes]):
        env.seed = seed
        state = env.reset()
        state_index = env.discretize_state(state)
        
        episode_reward = 0
        queue_lengths = []
        total_waiting_time = 0
        
        done = False
        step = 0
        
        while not done:
            # Usar política greedy (sin exploración)
            action = agent.select_action(state_index, training=False)
            
            # Contar acciones
            if action == 0:
                action_counts["maintain"] += 1
            else:
                action_counts["change"] += 1
            
            next_state, reward, done, info = env.step(action)
            next_state_index = env.discretize_state(next_state)
            
            episode_reward += reward
            queue_lengths.append(info["total_vehicles"])
            total_waiting_time += info["total_vehicles"]
            
            if render and i == 0:  # Renderizar solo el primer episodio
                env.render()
            
            state_index = next_state_index
            step += 1
        
        episode_rewards.append(episode_reward)
        episode_avg_queues.append(np.mean(queue_lengths))
        episode_max_queues.append(np.max(queue_lengths))
        episode_total_waiting.append(total_waiting_time)
    
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_avg_queue": np.mean(episode_avg_queues),
        "std_avg_queue": np.std(episode_avg_queues),
        "mean_max_queue": np.mean(episode_max_queues),
        "mean_total_waiting": np.mean(episode_total_waiting),
        "episode_rewards": episode_rewards,
        "episode_avg_queues": episode_avg_queues,
        "action_counts": action_counts
    }
    
    return metrics


def evaluate_fixed_controller(
    switch_interval: int = 20,
    num_episodes: int = 100,
    seeds: list = None
):
    """
    Evalúa el controlador de tiempo fijo.
    """
    if seeds is None:
        seeds = range(1000, 1000 + num_episodes)
    
    env = TrafficEnvironment(max_vehicles=40)
    controller = FixedTimeController(switch_interval=switch_interval)
    
    episode_rewards = []
    episode_avg_queues = []
    episode_max_queues = []
    episode_total_waiting = []
    
    for seed in seeds[:num_episodes]:
        env.seed = seed
        state = env.reset()
        controller.reset()
        
        episode_reward = 0
        queue_lengths = []
        total_waiting_time = 0
        
        done = False
        
        while not done:
            state_index = env.discretize_state(state)
            action = controller.select_action(state_index)
            
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            queue_lengths.append(info["total_vehicles"])
            total_waiting_time += info["total_vehicles"]
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_avg_queues.append(np.mean(queue_lengths))
        episode_max_queues.append(np.max(queue_lengths))
        episode_total_waiting.append(total_waiting_time)
    
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_avg_queue": np.mean(episode_avg_queues),
        "std_avg_queue": np.std(episode_avg_queues),
        "mean_max_queue": np.mean(episode_max_queues),
        "mean_total_waiting": np.mean(episode_total_waiting),
        "episode_rewards": episode_rewards,
        "episode_avg_queues": episode_avg_queues
    }
    
    return metrics


def detect_outliers(data, threshold=3.0):
    """
    Detecta outliers usando el método IQR (Interquartile Range).
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    outlier_indices = np.where(outlier_mask)[0]
    outlier_values = data[outlier_mask]
    clean_data = data[~outlier_mask]
    
    return outlier_indices, outlier_values, clean_data


def compare_policies(model_path: str, num_episodes: int = 100, save_dir: str = "results"):
    """
    Compara el agente entrenado con un controlador de tiempo fijo.
    """
    print("=" * 60)
    print("EVALUACIÓN Y COMPARACIÓN DE POLÍTICAS")
    print("=" * 60)
    print()
    
    # Cargar agente entrenado
    env = TrafficEnvironment(max_vehicles=40)
    agent = QLearningAgent(
        state_space_size=env.get_state_space_size(),
        action_space_size=env.get_action_space_size()
    )
    agent.load_model(model_path)
    print()
    
    # Generar semillas consistentes para comparación justa
    seeds = range(2000, 2000 + num_episodes)
    
    # Evaluar agente Q-Learning
    print(f"Evaluando agente Q-Learning en {num_episodes} episodios")
    rl_metrics = evaluate_agent(agent, num_episodes, seeds)
    
    # Detectar outliers en Q-Learning
    rl_outlier_idx, rl_outlier_vals, rl_clean_rewards = detect_outliers(
        np.array(rl_metrics["episode_rewards"]), threshold=2.5
    )
    
    if len(rl_outlier_idx) > 0:
        print(f"Detectados {len(rl_outlier_idx)} outliers en Q-Learning (episodios: {rl_outlier_idx.tolist()})")
        print(f"Valores outliers: {rl_outlier_vals.astype(int).tolist()}")
        rl_median_reward = np.median(rl_metrics["episode_rewards"])
        print(f"Recompensa promedio: {rl_metrics["mean_reward"]:.2f} +/- {rl_metrics["std_reward"]:.2f}")
        print(f"Recompensa mediana: {rl_median_reward:.2f} (más robusta a outliers)")
    else:
        print(f"Recompensa promedio: {rl_metrics["mean_reward"]:.2f} +/- {rl_metrics["std_reward"]:.2f}")
    
    print(f"Cola promedio: {rl_metrics["mean_avg_queue"]:.2f} +/- {rl_metrics["std_avg_queue"]:.2f}")
    print()
    
    # Evaluar controlador fijo
    print(f"Evaluando controlador de tiempo fijo en {num_episodes} episodios")
    fixed_metrics = evaluate_fixed_controller(switch_interval=20, num_episodes=num_episodes, seeds=seeds)
    print(f"Recompensa promedio: {fixed_metrics["mean_reward"]:.2f} +/- {fixed_metrics["std_reward"]:.2f}")
    print(f"Cola promedio: {fixed_metrics["mean_avg_queue"]:.2f} +/- {fixed_metrics["std_avg_queue"]:.2f}")
    print()
    
    # Calcular mejora (usando mediana para robustez)
    rl_median = np.median(rl_metrics["episode_rewards"])
    fixed_median = np.median(fixed_metrics["episode_rewards"])
    
    reward_improvement = ((rl_metrics["mean_reward"] - fixed_metrics["mean_reward"]) / 
                          abs(fixed_metrics["mean_reward"]) * 100)
    reward_improvement_median = ((rl_median - fixed_median) / abs(fixed_median) * 100)
    
    queue_reduction = ((fixed_metrics["mean_avg_queue"] - rl_metrics["mean_avg_queue"]) / 
                       fixed_metrics["mean_avg_queue"] * 100)
    
    print("=" * 60)
    print("RESULTADOS DE LA COMPARACIÓN")
    print("=" * 60)
    print(f"\nMejora en recompensa (promedio): {reward_improvement:+.2f}%")
    print(f"Mejora en recompensa (mediana, más robusta): {reward_improvement_median:+.2f}%")
    print(f"Reducción en cola promedio: {queue_reduction:+.2f}%")
    print(f"Reducción en tiempo de espera: {(fixed_metrics["mean_total_waiting"] - rl_metrics["mean_total_waiting"]):.0f} vehículos-paso")
    
    if len(rl_outlier_idx) > 0:
        print(f"\nNota: Se detectaron {len(rl_outlier_idx)} episodios con desempeño atípico.")
        print(f"Esto puede deberse a escenarios de tráfico extremos no vistos en entrenamiento.")
        print(f"Considera aumentar episodios de entrenamiento o variar más las semillas.")
    print()
    
    # Guardar resultados
    os.makedirs(save_dir, exist_ok=True)
    
    results = {
        "rl_agent": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in rl_metrics.items()},
        "fixed_controller": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                            for k, v in fixed_metrics.items()},
        "comparison": {
            "reward_improvement_percent": reward_improvement,
            "reward_improvement_median_percent": reward_improvement_median,
            "queue_reduction_percent": queue_reduction,
            "waiting_time_reduction": fixed_metrics["mean_total_waiting"] - rl_metrics["mean_total_waiting"]
        },
        "outliers": {
            "rl_outlier_episodes": rl_outlier_idx.tolist() if len(rl_outlier_idx) > 0 else [],
            "rl_outlier_rewards": rl_outlier_vals.tolist() if len(rl_outlier_idx) > 0 else [],
            "num_outliers": len(rl_outlier_idx)
        }
    }
    
    results_path = os.path.join(save_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Resultados guardados en: {results_path}")
    
    # Graficar comparación
    plot_comparison(rl_metrics, fixed_metrics, save_dir)
    
    # Análisis detallado de episodio
    analyze_detailed_episode(agent, save_dir)
    
    return results


def analyze_detailed_episode(agent, save_dir):
    """
    Simula un episodio completo y genera análisis detallado.
    """
    env = TrafficEnvironment(max_vehicles=40, seed=123)
    state = env.reset()
    state_index = env.discretize_state(state)
    
    # Estadísticas
    total_vehicles_history = []
    queues_history = {'N': [], 'S': [], 'E': [], 'O': []}
    
    for step in range(500):
        action = agent.select_action(state_index, training=False)
        next_state, reward, done, info = env.step(action)
        next_state_index = env.discretize_state(next_state)
        
        # Guardar historia
        total_vehicles_history.append(info['total_vehicles'])
        queues_history['N'].append(info['queues'][0])
        queues_history['S'].append(info['queues'][1])
        queues_history['E'].append(info['queues'][2])
        queues_history['O'].append(info['queues'][3])
        
        state_index = next_state_index
        
        if done:
            break
    
    # Total de vehículos en el tiempo
    plt.figure(figsize=(12, 6))
    steps = range(len(total_vehicles_history))
    total_array = np.array(total_vehicles_history)
    
    plt.plot(steps, total_vehicles_history, linewidth=2, color='blue', alpha=0.7, label='Total de vehículos')
    plt.axhline(y=40, color='red', linestyle='--', linewidth=2, label='Límite máximo (40 veh)')
    plt.fill_between(steps, 0, total_vehicles_history, alpha=0.3, color='blue')
    
    avg_vehicles = total_array.mean()
    plt.axhline(y=avg_vehicles, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    plt.text(250, avg_vehicles + 2, f'Promedio: {avg_vehicles:.1f} veh', 
            ha='center', fontsize=10, color='green', fontweight='bold')
    
    plt.xlabel('Paso de Simulación', fontsize=11)
    plt.ylabel('Total de Vehículos', fontsize=11)
    plt.title('Total de Vehículos en el Sistema a lo Largo del Tiempo', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 500)
    plt.ylim(0, 45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'analysis_total_vehicles.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Colas por dirección
    plt.figure(figsize=(12, 6))
    plt.plot(steps, queues_history['N'], label='Norte', linewidth=1.5)
    plt.plot(steps, queues_history['S'], label='Sur', linewidth=1.5)
    plt.plot(steps, queues_history['E'], label='Este', linewidth=1.5)
    plt.plot(steps, queues_history['O'], label='Oeste', linewidth=1.5)
    plt.xlabel('Paso de Simulación', fontsize=11)
    plt.ylabel('Vehículos en Cola', fontsize=11)
    plt.title('Distribución de Vehículos por Carril', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10, ncol=4, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 500)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'analysis_queues_by_lane.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráficos de análisis guardados en: {save_dir}/")
    print(f"  - analysis_total_vehicles.png")
    print(f"  - analysis_queues_by_lane.png")


def plot_comparison(rl_metrics, fixed_metrics, save_dir):
    """
    Genera gráficos comparativos entre agente y controlador fijo.
    """
    # Detectar outliers para visualización
    rl_outlier_idx, rl_outlier_vals, rl_clean = detect_outliers(
        np.array(rl_metrics["episode_rewards"]), threshold=2.5
    )
    
    # Distribución de recompensas
    plt.figure(figsize=(10, 6))
    plt.hist(rl_metrics["episode_rewards"], bins=30, alpha=0.7, label="Q-Learning", color="blue")
    plt.hist(fixed_metrics["episode_rewards"], bins=30, alpha=0.7, label="Tiempo Fijo", color="red")
    
    if len(rl_outlier_idx) > 0:
        outlier_range = [np.min(rl_outlier_vals), np.max(rl_outlier_vals)]
        plt.axvspan(outlier_range[0], outlier_range[1], alpha=0.2, color="yellow", 
                label=f"{len(rl_outlier_idx)} outliers detectados")
    
    plt.xlabel("Recompensa Total", fontsize=11)
    plt.ylabel("Frecuencia", fontsize=11)
    plt.title("Distribución de Recompensas", fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_rewards_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Boxplot de colas promedio
    plt.figure(figsize=(10, 6))
    data_to_plot = [rl_metrics["episode_avg_queues"], fixed_metrics["episode_avg_queues"]]
    bp = plt.boxplot(data_to_plot, tick_labels=["Q-Learning", "Tiempo Fijo"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightcoral")
    plt.ylabel("Vehículos Promedio en Cola", fontsize=11)
    plt.title("Comparación de Congestión", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_congestion_boxplot.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Distribución de decisiones del agente
    plt.figure(figsize=(10, 8))
    action_counts = rl_metrics["action_counts"]
    total_actions = action_counts["maintain"] + action_counts["change"]
    sizes = [action_counts["maintain"], action_counts["change"]]
    labels = [f"Mantener\n{action_counts['maintain']:,} acciones\n({100*action_counts['maintain']/total_actions:.1f}%)",
            f"Cambiar\n{action_counts['change']:,} acciones\n({100*action_counts['change']/total_actions:.1f}%)"]
    colors = ["#66b3ff", "#ff6b6b"]
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
                                        autopct="%1.1f%%", shadow=True, startangle=90, 
                                        textprops={"fontsize": 10})
    
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(11)
    
    plt.title("Distribución de Decisiones del Agente Q-Learning\n(Total: {:,} acciones)".format(total_actions), 
                fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_action_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Comparación de métricas clave
    plt.figure(figsize=(10, 6))
    metrics_names = ["Recompensa\nMedia", "Cola\nPromedio", "Cola\nMáxima"]
    rl_values = [
        rl_metrics["mean_reward"],
        rl_metrics["mean_avg_queue"],
        rl_metrics["mean_max_queue"]
    ]
    fixed_values = [
        fixed_metrics["mean_reward"],
        fixed_metrics["mean_avg_queue"],
        fixed_metrics["mean_max_queue"]
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, rl_values, width, label="Q-Learning", color="blue", alpha=0.7)
    bars2 = plt.bar(x + width/2, fixed_values, width, label="Tiempo Fijo", color="red", alpha=0.7)
    
    plt.ylabel("Valor", fontsize=11)
    plt.title("Comparación de Métricas Clave", fontsize=13, fontweight="bold")
    plt.xticks(x, metrics_names, fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis="y")
    
    # Añadir valores sobre las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f"{height:.1f}",
                    ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_key_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Evolución temporal
    plt.figure(figsize=(12, 6))
    episodes_to_show = min(20, len(rl_metrics["episode_rewards"]))
    x_episodes = range(episodes_to_show)
    
    rl_sample = rl_metrics["episode_rewards"][:episodes_to_show]
    fixed_sample = fixed_metrics["episode_rewards"][:episodes_to_show]
    
    plt.plot(x_episodes, rl_sample, marker="o", label="Q-Learning", 
            linewidth=2, color="blue", markersize=6, alpha=0.8)
    plt.plot(x_episodes, fixed_sample, marker="s", label="Tiempo Fijo", 
            linewidth=2, color="red", markersize=6, alpha=0.8)
    
    # Marcar outliers si los hay en esta muestra
    for idx in rl_outlier_idx:
        if idx < episodes_to_show:
            plt.plot(idx, rl_metrics["episode_rewards"][idx], "r*", 
                    markersize=15, markeredgewidth=2, markeredgecolor="darkred",
                    label="Outlier" if idx == rl_outlier_idx[0] else "", zorder=10)
    
    plt.xlabel("Episodio", fontsize=11)
    plt.ylabel("Recompensa Total", fontsize=11)
    plt.title(f"Evolución en Primeros {episodes_to_show} Episodios", fontsize=13, fontweight="bold")
    plt.xticks(range(episodes_to_show))
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_temporal_evolution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Gráficos de comparación guardados en: {save_dir}/")
    print(f"  - comparison_rewards_distribution.png")
    print(f"  - comparison_congestion_boxplot.png")
    print(f"  - comparison_action_distribution.png")
    print(f"  - comparison_key_metrics.png")
    print(f"  - comparison_temporal_evolution.png")


if __name__ == "__main__":
    # Configuración
    MODEL_PATH = "models/qlearning_traffic_agent.pkl"
    NUM_EPISODES = 100
    SAVE_DIR = "results"
    
    # Verificar que existe el modelo
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}")
        print("Por favor, ejecuta primero train.py para entrenar el agente.")
    else:
        # Comparar políticas
        results = compare_policies(MODEL_PATH, NUM_EPISODES, SAVE_DIR)
        
        print("\n Evaluación completada exitosamente")
