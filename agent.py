"""
Agente inteligente para control de semáforos usando Q-Learning.
Implementa el algoritmo de aprendizaje por refuerzo.
"""

import numpy as np
import pickle
import os
from typing import Optional


class QLearningAgent:
    """
    Agente que aprende una política óptima para control de semáforos
    usando el algoritmo Q-Learning.
    
    Q-Learning es un algoritmo off-policy que aprende la función Q(s, a)
    que representa el valor esperado de tomar la acción "a" en el estado "s".
    """
    
    def __init__(
        self,
        state_space_size: int,
        action_space_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        """
        Inicializa el agente Q-Learning.
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        
        # Hiperparámetros de Q-Learning
        self.alpha = learning_rate  # Tasa de aprendizaje
        self.gamma = discount_factor  # Factor de descuento
        self.epsilon = epsilon  # Probabilidad de exploración
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Tabla Q: Q(s, a) para todos los pares estado-acción
        self.q_table = np.zeros((state_space_size, action_space_size))
        
        # Estadísticas de entrenamiento
        self.training_stats = {
            "episodes": 0,
            "total_steps": 0,
            "epsilon_history": []
        }
    
    def select_action(self, state_index: int, training: bool = True) -> int:
        """
        Selecciona una acción usando política epsilon-greedy.
        """
        if training and np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return np.random.randint(self.action_space_size)
        else:
            # Explotación: mejor acción según Q-table
            return np.argmax(self.q_table[state_index])
    
    def update(
        self,
        state_index: int,
        action: int,
        reward: float,
        next_state_index: int,
        done: bool
    ):
        """
        Actualiza la Q-table usando la ecuación de Q-Learning.
        """
        # Valor actual de Q(s,a)
        current_q = self.q_table[state_index, action]
        
        # Valor máximo de Q(s',a')
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state_index])
        
        target = reward + self.gamma * max_next_q
        
        td_error = target - current_q
        
        # Actualización de Q-Learning
        new_q = current_q + self.alpha * td_error
        self.q_table[state_index, action] = new_q
        
        # Actualizar estadísticas
        self.training_stats["total_steps"] += 1
    
    def decay_epsilon(self):
        """
        Reduce epsilon después de cada episodio (exploración → explotación).
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_stats["epsilon_history"].append(self.epsilon)
    
    def finish_episode(self):
        """
        Marca el fin de un episodio de entrenamiento.
        """
        self.training_stats["episodes"] += 1
        self.decay_epsilon()
    
    def save_model(self, filepath: str):
        """
        Guarda el modelo entrenado (Q-table y parámetros).
        """
        model_data = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "training_stats": self.training_stats,
            "hyperparameters": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Carga un modelo previamente entrenado.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
        
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data["q_table"]
        self.epsilon = model_data["epsilon"]
        self.training_stats = model_data["training_stats"]
        
        # Cargar hiperparámetros si existen
        if "hyperparameters" in model_data:
            hyper = model_data["hyperparameters"]
            self.alpha = hyper["alpha"]
            self.gamma = hyper["gamma"]
            self.epsilon_decay = hyper["epsilon_decay"]
            self.epsilon_min = hyper["epsilon_min"]
        
        print(f"Modelo cargado desde: {filepath}")
        print(f"Episodios entrenados: {self.training_stats["episodes"]}")
        print(f"Epsilon actual: {self.epsilon:.4f}")
    
    def get_q_values(self, state_index: int) -> np.ndarray:
        """
        Retorna los valores Q para todas las acciones en un estado dado.
        """
        return self.q_table[state_index].copy()
    
    def get_policy(self) -> np.ndarray:
        """
        Extrae la política óptima aprendida (acción con mayor Q en cada estado).
        """
        return np.argmax(self.q_table, axis=1)
    
    def get_stats(self) -> dict:
        """
        Retorna estadísticas de entrenamiento.
        """
        return self.training_stats.copy()
