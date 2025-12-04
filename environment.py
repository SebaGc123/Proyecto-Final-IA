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
    
    Estados: (cola_norte, cola_sur, cola_este, cola_oeste, fase_actual)
    Acciones: 0=Mantener, 1=Cambiar
    Recompensas: -suma de vehículos esperando en luz roja
    """
    
    def __init__(self, max_vehicles: int = 40, seed: int = None):
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
        
        # Tiempo de cambio de fase (amarillo + rojo total)
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
        
        # Capacidad máxima de procesamiento por paso (vehículos que cruzan)
        self.throughput_rate = 2  # vehículos que cruzan por paso si luz verde
        
    def _generate_arrival_probabilities(self) -> np.ndarray:
        """
        Genera probabilidades de llegada asimétricas para cada dirección.
        """
        # Probabilidades base
        probs = self.rng.uniform(0.1, 0.4, size=self.num_lanes)
        # Normalizar para que la suma sea razonable
        probs = probs / probs.sum() * 0.3  # Ajustar escala
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
        
        # Generar distribución inicial de vehículos
        initial_vehicles = self.rng.randint(0, self.max_vehicles)
        self.queues = self._distribute_vehicles(initial_vehicles)
        
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
        """
        return np.append(self.queues.copy(), self.current_phase)
    
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
        Norte-Sur (fase 0) o Este-Oeste (fase 1).
        """
        if self.in_transition:
            # Durante transición, nadie cruza
            return
        
        if self.current_phase == 0:  # Norte-Sur verde
            green_lanes = [0, 1]  # Norte y Sur
        else:  # Este-Oeste verde
            green_lanes = [2, 3]  # Este y Oeste
        
        # Procesar vehículos en carriles verdes
        for lane in green_lanes:
            vehicles_to_process = min(self.queues[lane], self.throughput_rate)
            self.queues[lane] -= vehicles_to_process
    
    def _calculate_reward(self) -> float:
        """
        Calcula la recompensa basada en vehículos esperando en luz roja.
        """
        if self.current_phase == 0:  # Norte-Sur verde
            red_lanes = [2, 3]  # Este y Oeste en rojo
        else:  # Este-Oeste verde
            red_lanes = [0, 1]  # Norte y Sur en rojo
        
        # Suma de vehículos en luz roja (penalización)
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
        done = self.time_step >= 500  # Episodio de 500 pasos
        
        # Información adicional
        info = {
            'total_vehicles': self.queues.sum(),
            'phase': self.current_phase,
            'in_transition': self.in_transition,
            'queues': self.queues.copy()
        }
        
        return new_state, reward, done, info
    
    def get_state_space_size(self, carriles=4, niveles=4, fases=2) -> int:
        """
        Retorna el tamaño del espacio de estados.
        Para discretización en Q-Learning.
        """
        # [0-5, 6-10, 11-15, 16+] = 4 niveles por carril
        # 4 carriles * 4 niveles = 4^4 = 256 estados de colas
        estados_colas = niveles ** carriles
        # * 2 fases = 512 estados totales
        estados_totales = estados_colas * fases
        return estados_totales
    
    def get_action_space_size(self, acciones=2) -> int:
        """Retorna el número de acciones posibles."""
        # Mantener o Cambiar
        return acciones
    
    def discretize_state(self, state: np.ndarray) -> int:
        """
        Convierte un estado continuo a un índice discreto.
        """
        queues = state[:4]
        phase = int(state[4])
        
        # Discretizar colas en 4 niveles: 0-5, 6-10, 11-15, 16+
        discretized_queues = []
        for q in queues:
            if q <= 5:
                level = 0
            elif q <= 10:
                level = 1
            elif q <= 15:
                level = 2
            else:
                level = 3
            discretized_queues.append(level)
        
        # Convertir a índice único
        # Usar base-4 para las colas (4 niveles) y multiplicar por fase
        index = 0
        for i, level in enumerate(discretized_queues):
            index += level * (4 ** i)
        
        # Añadir fase (multiplica por 2 el espacio)
        index = index * 2 + phase
        
        return index
