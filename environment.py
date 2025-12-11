"""
Entorno de simulación de intersección con 4 vías para control de semáforos.
Implementa un proceso de decisión de Markov (MDP) estocástico.
"""

import numpy as np
from typing import Tuple, Dict
import random


class TrafficEnvironment:
    """
    Simulador de una intersección de 4 vías (Norte, Sur, Este, Oeste).
    """
    
    def __init__(self, max_vehicles: int = 100, seed: int = None):
        """
        Inicializa el entorno.
        """
        self.max_vehicles = max_vehicles
        self.seed = seed
        
        # Direcciones: 0=Norte, 1=Sur, 2=Este, 3=Oeste
        self.num_lanes = 4
        
        # Fases del semáforo: 0=Norte-Sur (verde), 1=Este-Oeste (verde)
        self.current_phase = 0
        
        # Colas de vehículos en cada dirección
        self.queues = np.zeros(self.num_lanes, dtype=int)
        
        # Contador de pasos de tiempo
        self.time_step = 0
        
        # Tiempo mínimo antes de cambiar fase
        self.min_phase_duration = 5
        self.steps_in_current_phase = 0
        
        # Tiempo de cambio de fase
        self.transition_time = 2
        self.in_transition = False
        self.transition_counter = 0
        
        # Generador de números aleatorios
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        
        # Probabilidades de llegada de vehículos
        self.arrival_probs = self._generate_arrival_probabilities()
        
        # Sistema de progreso de cruce
        self.crossing_steps = 5  # pasos necesarios para cruzar
        self.crossing_progress = {lane: [] for lane in range(self.num_lanes)}  # vehículos cruzando
        self.throughput_rate = 4  # vehículos cruzando simultáneamente si luz verde
        
    def _generate_arrival_probabilities(self) -> np.ndarray:
        """
        Genera probabilidades de llegada asimétricas para cada dirección.
        """
        # Probabilidades balanceadas para crear tráfico manejable
        probs = self.rng.uniform(0.2, 0.5, size=self.num_lanes)
        probs = probs / probs.sum() * 0.45
        return probs
    
    def reset(self) -> np.ndarray:
        """
        Reinicia el entorno a un estado inicial.
        """
        # Resetear fase a Norte-Sur
        self.current_phase = 0
        self.steps_in_current_phase = 0
        self.in_transition = False
        self.transition_counter = 0
        self.time_step = 0
        
        # Inicializar con 10 vehículos por carril
        self.queues = np.array([10, 10, 10, 10], dtype=int)
        
        # Resetear progreso de cruce
        self.crossing_progress = {lane: [] for lane in range(self.num_lanes)}
        
        # Generar nuevas probabilidades de llegada
        self.arrival_probs = self._generate_arrival_probabilities()
        
        return self._get_state()
    
    def _distribute_vehicles(self, total: int) -> np.ndarray:
        """
        Distribuye un número total de vehículos aleatoriamente entre las 4 vías.
        """
        if total == 0:
            return np.zeros(self.num_lanes, dtype=int)
        
        # Usar distribución multinomial para distribución más realista
        probs = self.rng.dirichlet(np.ones(self.num_lanes))
        distribution = self.rng.multinomial(total, probs)
        return distribution.astype(int)
    
    def _get_state(self) -> np.ndarray:
        """
        Retorna el estado actual del entorno.
        Incluye colas, fase actual y pasos en la fase actual.
        """
        return np.append(self.queues.copy(), [self.current_phase, self.steps_in_current_phase])
    
    def _generate_new_vehicles(self):
        """
        Genera nuevos vehículos que llegan a la intersección usando probabilidades.
        """
        total_current = self.queues.sum()
        
        for lane in range(self.num_lanes):
            # Probabilidad de que llegue un vehículo
            if self.rng.random() < self.arrival_probs[lane]:
                if total_current < self.max_vehicles:
                    self.queues[lane] += 1
                    total_current += 1
    
    def _process_green_lanes(self):
        """
        Procesa los vehículos en las direcciones con luz verde.
        """
        if self.in_transition:
            # Durante transición, nadie cruza ni avanza
            return
        
        if self.current_phase == 0:
            green_lanes = [0, 1]  # Norte y Sur
        else:
            green_lanes = [2, 3]  # Este y Oeste
        
        # Procesar vehículos en carriles verdes
        for lane in green_lanes:
            # Avanzar el progreso de los vehículos que ya están cruzando
            still_crossing = []
            for progress in self.crossing_progress[lane]:
                progress += 1
                if progress < self.crossing_steps:
                    still_crossing.append(progress)
            self.crossing_progress[lane] = still_crossing
            
            # Iniciar el cruce de nuevos vehículos
            space_available = self.throughput_rate - len(self.crossing_progress[lane])
            vehicles_to_start = min(self.queues[lane], max(0, space_available))
            
            for _ in range(vehicles_to_start):
                self.queues[lane] -= 1
                self.crossing_progress[lane].append(0)
    
    def _calculate_reward(self) -> float:
        """
        Calcula la recompensa basada en vehículos esperando en luz roja.
        """
        if self.current_phase == 0:
            red_lanes = [2, 3]
        else:
            red_lanes = [0, 1]
        
        # Suma de vehículos en luz roja
        waiting_vehicles = sum(self.queues[lane] for lane in red_lanes)
        
        # También penalizar ligeramente los vehículos en verde que aún esperan
        green_penalty = sum(self.queues) * 0.1
        
        reward = -(waiting_vehicles + green_penalty)
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Ejecuta una acción en el entorno.
        """
        self.time_step += 1
        self.steps_in_current_phase += 1
        
        # Procesar transición si está activa
        if self.in_transition:
            self.transition_counter += 1
            if self.transition_counter >= self.transition_time:
                # Completar transición
                self.current_phase = 1 - self.current_phase
                self.in_transition = False
                self.transition_counter = 0
                self.steps_in_current_phase = 0
        else:
            # Procesar acción si no estamos en transición
            if action == 1 and self.steps_in_current_phase >= self.min_phase_duration:
                # Iniciar cambio de fase
                self.in_transition = True
                self.transition_counter = 0
        
        # Procesar vehículos en luz verde
        self._process_green_lanes()
        
        # Generar nuevos vehículos
        self._generate_new_vehicles()
        
        # Calcular recompensa
        reward = self._calculate_reward()
        
        # Obtener nuevo estado
        new_state = self._get_state()
        
        # El episodio termina después de cierto número de pasos
        done = self.time_step >= 500
        
        info = {
            "total_vehicles": self.queues.sum(),
            "phase": self.current_phase,
            "in_transition": self.in_transition,
            "queues": self.queues.copy()
        }
        
        return new_state, reward, done, info
    
    def get_state_space_size(self) -> int:
        """
        Retorna el tamaño del espacio de estados.
        Para discretización en Q-Learning.
        """
        # Estado: diferencia de colas (25 niveles) + fase (2) + tiempo en fase (4 niveles)
        # 25 * 2 * 4 = 200 estados totales
        return 200
    
    def get_action_space_size(self, acciones=2) -> int:
        """Retorna el número de acciones posibles."""
        # Mantener o Cambiar
        return acciones
    
    def discretize_state(self, state: np.ndarray) -> int:
        """
        Convierte un estado continuo a un índice discreto.
        Usa la DIFERENCIA entre Norte-Sur y Este-Oeste.
        Incluye tiempo en fase actual para decidir cuándo cambiar.
        """
        queues = state[:4]
        phase = int(state[4])
        steps_in_phase = int(state[5]) if len(state) > 5 else 0
        
        # Calcular presión en cada dirección
        ns_pressure = queues[0] + queues[1]
        ew_pressure = queues[2] + queues[3]
        
        # Diferencia de presión (-100 a +100)
        diff = ns_pressure - ew_pressure
        
        # Discretizar diferencia en 25 niveles
        if diff == 0:
            level = 11
        else:
            neg_thresholds = [-40, -30, -20, -15, -10, -8, -6, -4, -2, -1, 0]
            pos_thresholds = [1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60]
            
            if diff < 0:
                level = 0
                for i, threshold in enumerate(neg_thresholds):
                    if diff < threshold:
                        level = i
                        break
            else:
                level = 12
                for i, threshold in enumerate(pos_thresholds):
                    if diff <= threshold:
                        level = 12 + i
                        break
                else:
                    level = 24

        if steps_in_phase < 8:
            time_level = 0
        elif steps_in_phase < 16:
            time_level = 1
        elif steps_in_phase < 26:
            time_level = 2
        else:
            time_level = 3
            
        index = (level * 2 + phase) * 4 + time_level
        
        return index
