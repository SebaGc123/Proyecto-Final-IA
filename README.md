# Control Inteligente de Semáforos con Q-Learning

## Requisitos

### Software Necesario
- **Python 3.12** o superior
- **NumPy** - Para cálculos numéricos y matrices
- **Matplotlib** - Para visualización y gráficos

### Instalación de Dependencias

```powershell
pip install -r requirements.txt
```

---

## Estructura del Proyecto

```
IA_Semaforos/
├── environment.py          # Entorno de simulación de tráfico (MDP)
├── agent.py               # Agente Q-Learning
├── train.py               # Script de entrenamiento
├── evaluate.py            # Evaluación y comparación
├── visualize.py           # Visualizador interactivo con controles
├── requirements.txt       # Dependencias del proyecto
└── README.md             # Este archivo
```

### Descripción de Archivos

- **environment.py**: Implementa el entorno de simulación de la intersección con 4 direcciones (N-S-E-O). Maneja la llegada de vehículos, cambios de fase del semáforo, cálculo de recompensas y discretización de estados basada en presión diferencial.

- **agent.py**: Implementa el agente Q-Learning que aprende la política óptima de control. Contiene la tabla Q (200×2), política ε-greedy y actualización de valores.

- **train.py**: Ejecuta el proceso de entrenamiento por 2000 episodios. Guarda el modelo entrenado, métricas en JSON y genera 4 gráficos de análisis del proceso de aprendizaje.

- **evaluate.py**: Compara el rendimiento del agente entrenado contra una política de tiempo fijo. Ejecuta 100 episodios de prueba y genera 7 gráficos de análisis detallado.

- **visualize.py**: Visualizador interactivo en tiempo real con controles de pausa/reproducción y avance por pasos. Muestra el estado de la intersección, vehículos en cada dirección y métricas actuales.

---

## Gráficos Generados

### Entrenamiento (train.py)
El script genera **4 archivos PNG** en la carpeta models:

1. **training_rewards.png**: Evolución de la recompensa acumulada por episodio
2. **training_avg_queue.png**: Longitud promedio de colas por episodio
3. **training_epsilon.png**: Decaimiento del factor de exploración (ε)
4. **training_actions.png**: Distribución de acciones (mantener vs cambiar fase)

### Evaluación (evaluate.py)
El script genera **7 archivos PNG** en la carpeta results:

1. **comparison_rewards.png**: Comparación de recompensas (RL vs Tiempo Fijo)
2. **comparison_avg_queue.png**: Comparación de colas promedio
3. **comparison_waiting_time.png**: Comparación de tiempo de espera acumulado
4. **comparison_max_queue.png**: Comparación de cola máxima observada
5. **comparison_throughput.png**: Comparación de vehículos procesados
6. **rl_action_distribution.png**: Distribución de acciones del agente RL
7. **rl_queue_evolution.png**: Evolución de colas por dirección (N-S-E-O)

---

## Cómo Ejecutar el Programa

### Orden de Ejecución Recomendado

#### 1. Entrenamiento del Agente
```powershell
python train.py
```

**Qué hace:**
- Entrena el agente Q-Learning por 2000 episodios
- Guarda el modelo en: `qlearning_traffic_agent.pkl`
- Guarda métricas en: `training_metrics.json`
- Genera 4 gráficos PNG del proceso de entrenamiento

#### 2. Evaluación del Desempeño
```powershell
python evaluate.py
```

**Qué hace:**
- Carga el modelo entrenado
- Compara contra política de tiempo fijo en 100 episodios
- Guarda resultados en: `evaluation_results.json`
- Genera 7 gráficos PNG de análisis comparativo

**Nota:** Requiere que primero se haya ejecutado `train.py` para tener el modelo guardado.

#### 3. Visualización Interactiva
```powershell
python visualize.py
```

**Qué hace:**
- Abre una ventana interactiva mostrando la simulación en tiempo real
- Muestra vehículos, semáforos, y métricas actuales
- Permite pausar/reanudar y avanzar paso a paso

**Controles:**
- **Botón Pausa/Reanudar**: Detiene o continúa la simulación
- **Botón Avanzar Paso**: Ejecuta un solo paso (solo disponible en pausa)
- **Cerrar ventana**: Termina la visualización
