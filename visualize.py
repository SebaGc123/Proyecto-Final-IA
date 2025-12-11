"""
Visualizador interactivo del comportamiento del agente.
Muestra una simulación visual de la intersección y el control del semáforo.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.animation as animation
from matplotlib.widgets import Button
from environment import TrafficEnvironment
from agent import QLearningAgent
import os
import time


class TrafficVisualizer:
    """
    Visualizador gráfico de la intersección de tráfico.
    """
    def __init__(self, env, agent=None, interval=200):
        self.env = env
        self.agent = agent
        self.interval = interval
        
        # Configurar figura con espacio para controles
        self.fig = plt.figure(figsize=(8, 8))
        self.ax_main = self.fig.add_axes([0.1, 0.2, 0.8, 0.75])
        
        self.time_step = 0
        self.paused = False
        self.anim = None
        self.stepping = False
        self.last_step_time = 0
        
        # Crear controles interactivos
        self._create_controls()
    
    def _create_controls(self):
        """Crea los botones y controles interactivos."""
        # Botón Play/Pause
        ax_pause = self.fig.add_axes([0.3, 0.08, 0.15, 0.05])
        self.btn_pause = Button(ax_pause, "Pausar", color="lightcoral", hovercolor="salmon")
        self.btn_pause.on_clicked(self._toggle_pause)
        
        # Botón Step
        ax_step = self.fig.add_axes([0.55, 0.08, 0.15, 0.05])
        self.btn_step = Button(ax_step, "Avanzar Paso", color="lightgray", hovercolor="silver")
        self.btn_step.on_clicked(self._step_forward)
        self.btn_step.ax.set_alpha(0.5)
        
        # Etiqueta de estado
        self.status_text = self.fig.text(
            0.5, 0.14, 
            "Estado: Reproduciendo", 
            ha="center", 
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8)
        )
    
    def _toggle_pause(self, event):
        """Alterna entre pausa y reproducción."""
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.label.set_text("Reanudar")
            self.btn_pause.color = "lightgreen"
            self.status_text.set_text("Estado: Pausado")
            self.status_text.set_bbox(dict(boxstyle="round", facecolor="yellow", alpha=0.8))
            # Habilitar botón de avanzar paso
            self.btn_step.ax.set_alpha(1.0)
            self.btn_step.color = "lightblue"
        else:
            self.btn_pause.label.set_text("Pausar")
            self.btn_pause.color = "lightcoral"
            self.status_text.set_text("Estado: Reproduciendo")
            self.status_text.set_bbox(dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))
            # Deshabilitar botón de avanzar paso
            self.btn_step.ax.set_alpha(0.5)
            self.btn_step.color = "lightgray"
        self.fig.canvas.draw_idle()
    def _step_forward(self, event):
        """Avanza un paso cuando está pausado."""
        if not self.paused or self.stepping:
            return
        
        current_time = time.time()
        if current_time - self.last_step_time < 0.1:
            return
        
        self.last_step_time = current_time
        
        if self.anim is not None:
            try:
                self.stepping = True
                state = self.env._get_state()
                state_index = self.env.discretize_state(state)
                
                if self.agent is not None:
                    action = self.agent.select_action(state_index, training=False)
                else:
                    action = np.random.randint(2)
                
                next_state, reward, done, info = self.env.step(action)
                self.time_step += 1
                
                # Redibujar
                self.setup_main_axes()
                self.draw_vehicles(info["queues"])
                self.draw_traffic_lights(info["phase"], info["in_transition"])
                
                total_vehicles = info["total_vehicles"]
                phase_name = "Norte-Sur" if info["phase"] == 0 else "Este-Oeste"
                action_name = "Mantener" if action == 0 else "Cambiar"
                
                info_text = f"Paso: {self.time_step}\nTotal: {total_vehicles} vehículos\nFase: {phase_name}\nAcción: {action_name}"
                
                self.ax_main.text(-14.5, 14, info_text, 
                                fontsize=10, 
                                verticalalignment="top",
                                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
                
                if done:
                    print("Episodio terminado")
                    self.status_text.set_text("Estado: Episodio Terminado")
                    self.status_text.set_bbox(dict(boxstyle="round", facecolor="lightcoral", alpha=0.8))
                
                self.fig.canvas.draw_idle()
            finally:
                # Asegurar que stepping siempre se resetee
                self.stepping = False
        
    def setup_main_axes(self):
        """Configura el panel principal de visualización."""
        self.ax_main.clear()
        self.ax_main.set_xlim(-15, 15)
        self.ax_main.set_ylim(-15, 15)
        self.ax_main.set_aspect("equal")
        self.ax_main.axis("off")
        
        # Dibujar calles
        # Horizontal
        self.ax_main.add_patch(Rectangle((-15, -3), 30, 6, color="gray", alpha=0.3))
        # Vertical
        self.ax_main.add_patch(Rectangle((-3, -15), 6, 30, color="gray", alpha=0.3))
        
        # Líneas de división
        self.ax_main.plot([-15, -3], [0, 0], "w--", linewidth=1, alpha=0.5)
        self.ax_main.plot([3, 15], [0, 0], "w--", linewidth=1, alpha=0.5)
        self.ax_main.plot([0, 0], [-15, -3], "w--", linewidth=1, alpha=0.5)
        self.ax_main.plot([0, 0], [3, 15], "w--", linewidth=1, alpha=0.5)
        
        # Etiquetas de direcciones
        self.ax_main.text(0, 12, "NORTE", ha="center", fontsize=12, fontweight="bold")
        self.ax_main.text(0, -12, "SUR", ha="center", fontsize=12, fontweight="bold")
        self.ax_main.text(12, 0, "ESTE", ha="center", fontsize=12, fontweight="bold")
        self.ax_main.text(-12, 0, "OESTE", ha="center", fontsize=12, fontweight="bold")
        
    def draw_vehicles(self, queues):
        """
        Dibuja los vehículos en cada carril.
        """
        vehicle_size = 0.8
        spacing = 1.2
        
        # Norte - carril izquierdo
        for i in range(min(queues[0], 8)):  # Máximo 8 visibles
            y = 4 + i * spacing
            self.ax_main.add_patch(Rectangle((-1.5, y), vehicle_size, vehicle_size, 
                                            color="blue", alpha=0.7))
        
        # Sur - carril derecho
        for i in range(min(queues[1], 8)):
            y = -4 - i * spacing
            self.ax_main.add_patch(Rectangle((0.7, y), vehicle_size, vehicle_size, 
                                            color="blue", alpha=0.7))
        
        # Este - carril superior
        for i in range(min(queues[2], 8)):
            x = 4 + i * spacing
            self.ax_main.add_patch(Rectangle((x, 0.7), vehicle_size, vehicle_size, 
                                            color="blue", alpha=0.7))
        
        # Oeste - carril inferior
        for i in range(min(queues[3], 8)):
            x = -4 - i * spacing
            self.ax_main.add_patch(Rectangle((x, -1.5), vehicle_size, vehicle_size, 
                                            color="blue", alpha=0.7))
        
        # Mostrar números si hay más vehículos
        if queues[0] > 8:
            self.ax_main.text(-1, 12, f"+{queues[0]-8}", fontsize=10, 
                            ha="center", fontweight="bold")
        if queues[1] > 8:
            self.ax_main.text(1.2, -12, f"+{queues[1]-8}", fontsize=10, 
                            ha="center", fontweight="bold")
        if queues[2] > 8:
            self.ax_main.text(12, 1.2, f"+{queues[2]-8}", fontsize=10, 
                            ha="center", fontweight="bold")
        if queues[3] > 8:
            self.ax_main.text(-12, -1, f"+{queues[3]-8}", fontsize=10, 
                            ha="center", fontweight="bold")
    
    def draw_traffic_lights(self, phase, in_transition):
        """
        Dibuja los semáforos.
        """
        light_radius = 0.5
        
        # Determinar colores
        if in_transition:
            ns_color = "yellow"
            ew_color = "yellow"
        elif phase == 0:  # Norte-Sur verde
            ns_color = "green"
            ew_color = "red"
        else:  # Este-Oeste verde
            ns_color = "red"
            ew_color = "green"
        
        # Semáforos Norte-Sur
        self.ax_main.add_patch(Circle((2.5, 3.5), light_radius, color=ns_color, ec="black", linewidth=2))
        self.ax_main.add_patch(Circle((-2.5, -3.5), light_radius, color=ns_color, ec="black", linewidth=2))
        
        # Semáforos Este-Oeste
        self.ax_main.add_patch(Circle((3.5, -2.5), light_radius, color=ew_color, ec="black", linewidth=2))
        self.ax_main.add_patch(Circle((-3.5, 2.5), light_radius, color=ew_color, ec="black", linewidth=2))
    
    def animate_step(self, frame):
        """Función de animación para cada frame."""
        if self.paused:
            return []
        
        # Seleccionar acción
        state = self.env._get_state()
        state_index = self.env.discretize_state(state)
        
        if self.agent is not None:
            action = self.agent.select_action(state_index, training=False)
        else:
            action = np.random.randint(2)
        
        # Ejecutar paso
        next_state, reward, done, info = self.env.step(action)
        
        # Actualizar contador
        self.time_step += 1
        
        # Redibujar
        self.setup_main_axes()
        self.draw_vehicles(info["queues"])
        self.draw_traffic_lights(info["phase"], info["in_transition"])
        
        # Mostrar información
        total_vehicles = info["total_vehicles"]
        phase_name = "Norte-Sur" if info["phase"] == 0 else "Este-Oeste"
        action_name = "Mantener" if action == 0 else "Cambiar"
        
        info_text = f"Paso: {self.time_step}\nTotal: {total_vehicles} vehículos\nFase: {phase_name}\nAcción: {action_name}"
        
        self.ax_main.text(-14.5, 14, info_text, 
                        fontsize=10, 
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        
        if done:
            print("Episodio terminado")
            self.paused = True
            self.btn_pause.label.set_text("Reanudar")
            self.status_text.set_text("Estado: Episodio Terminado")
            self.status_text.set_bbox(dict(boxstyle="round", facecolor="lightcoral", alpha=0.8))
            return []
        
        return []
    
    def run(self, num_steps=500):
        """
        Ejecuta la visualización animada.
        """
        self.env.reset()
        self.setup_main_axes()
        
        self.anim = animation.FuncAnimation(
            self.fig, 
            self.animate_step, 
            frames=num_steps,
            interval=self.interval, 
            blit=False,
            repeat=False
        )
        
        plt.show()
        
        return self.anim


def visualize_agent(model_path: str, num_steps: int = 300, seed: int = 42):
    """
    Visualiza el comportamiento del agente entrenado.
    """
    print("=" * 60)
    print("VISUALIZACIÓN DEL AGENTE")
    print("=" * 60)
    print()
    
    # Crear entorno
    env = TrafficEnvironment(max_vehicles=40, seed=seed)
    
    # Cargar agente
    print("Cargando agente entrenado...")
    agent = QLearningAgent(
        state_space_size=env.get_state_space_size(),
        action_space_size=env.get_action_space_size()
    )
    agent.load_model(model_path)
    print()
    
    # Crear visualizador
    print(f"Iniciando visualización de {num_steps} pasos...")
    print("Cierra la ventana para terminar.\n")
    
    visualizer = TrafficVisualizer(env, agent, interval=100)
    visualizer.run(num_steps=num_steps)


if __name__ == "__main__":
    MODEL_PATH = "models/qlearning_traffic_agent.pkl"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}")
        print("Por favor, ejecuta primero train.py para entrenar el agente.")
    else:
        visualize_agent(MODEL_PATH, num_steps=300, seed=123)
