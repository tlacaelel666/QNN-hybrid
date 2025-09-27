<img width="980" height="250" alt="qnnimage" src="https://github.com/user-attachments/assets/9e703b44-2bb6-46ae-a91b-6a2160489ae6" />

# ⚛️ QNN-hybrid by QuoreMind: Advanced Quantum-Classical Framework (QNN, QEC & Bayesian Logic)
[![SmokApp](https://img.shields.io/badge/SmokApp-Software-black)](https://github.com/tlacaelel666/)
[![Estado de la Build](https://img.shields.io/badge/build-passing-brightgreen)](tlacaelel666/QNN-hybrid)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.5.0-orange)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24.2-blue)](https://scikit-learn.org/)
[![Python](https://img.shields.io/badge/Python-3.8-blue)](https://www.python.org/)
[![Licencia](https://img.shields.io/badge/license-Apache2.0-blue.svg)](LICENSE)
[![Contribuidores](https://img.shields.io/github/contributors/tlacaelel666/QNN-hybrid)](https://github.com/tlacaelel666/QNN-hybrid/graphs/contributors)
[![Último Commit](https://img.shields.io/github/last-commit/tlacaelel666/QNN-hybrid)](https://github.com/tlacaelel666/QNN-hybrid/commits/main)

## Visión General

**QuoreMind** es un *framework* modular de alto nivel diseñado para la implementación de algoritmos de **Aprendizaje Automático Cuántico (QML)**, con un enfoque crucial en la **resiliencia** y la **auditabilidad física** en entornos de *backend* ruidosos.

Este sistema hibrida la potencia del **Parameter-Shift Rule** para el entrenamiento con una arquitectura dual de Red Neuronal Cuántica (QNN) y le añade capas de seguridad y validación únicas: la **Corrección de Errores Cuánticos (QEC)** dinámica y una **Lógica Bayesiana de Falsabilidad** basada en la **Distancia de Mahalanobis**.

Es el *backend* ideal para desarrollar aplicaciones de **SaaS cuántico** y protocolos de seguridad que requieren justificación física y mitigación activa de ruido.

-----

## 🔑 Características Clave

| Módulo | Descripción | Principio Físico |
| :--- | :--- | :--- |
| **QNN Core** | Arquitectura cuántica doble (Circuitos A y B, 10 qubits máx.) con entrelazamiento inter-circuito. | **Entropía de Von Neumann** para medir el entrelazamiento. |
| **Estabilidad** | Implementación del **Parameter-Shift Rule**  para obtener gradientes analíticos (exactos) que mitigan el problema de la Meseta Estéril. | Cálculo de **Derivadas Exactas** (Parameter-Shift). |
| **Resiliencia (QEC)** | Simulación de dinámica de error mediante la **Ecuación de Lindblad** con aplicación periódica y optimizada de códigos QEC (e.g., Bit-Flip, Shor). | **Ecuación de Lindblad** (Dinámica de Sistemas Abiertos) y **Teoría QEC**. |
| **Falsabilidad & Seguridad** | Lógica de inferencia bayesiana que utiliza la **Distancia de Mahalanobis**  para auditar la desviación de los parámetros respecto a un estado *prior* físicamente deseado. | **Teorema de Bayes** y **Distancia de Mahalanobis** (para detección de anomalías). |
| **Persistencia** | `DatabaseManager` integrado con SQLite para el *logging* exhaustivo de métricas de entrenamiento (pérdida, precisión, entropía, norma del gradiente) | Gestión de Experimentos y Trazabilidad. |

-----

## 🛠️ Arquitectura Modular

El *framework* se compone de tres módulos principales que interactúan a través de un orquestador central (`main.py`):

1.  ### `AdvancedQNN`

      * **Propósito:** Implementa el circuito variacional y el *forward/backward pass*.
      * **Componentes:** `QNNConfig` (configuración centralizada de capas, optimizadores y regularización L1/L2), `QNNParams` (contenedor de parámetros entrenables).
      * **Optimización:** Soporte para **SGD, Adam, RMSprop** con Parameter-Shift.

2.  ### `QuantumErrorCorrectionSimulator`

      * **Propósito:** Modela el ruido del *hardware* y aplica protocolos de corrección.
      * **Flujo:** El simulador evoluciona el estado cuántico bajo un `ErrorModel` (ej. Depolarizing, Bit-Flip) y, en intervalos óptimos, ejecuta los circuitos de corrección.

3.  ### `QuantumBayesMahalanobis`

      * **Propósito:** Proporciona la capa de auditoría y toma de decisiones para los protocolos de seguridad.
      * **Método:** Calcula la **Distancia de Mahalanobis** para determinar la "anomalía" o "desviación" de un estado cuántico con respecto a un *prior* (conjunto de estados "saludables") y utiliza esta métrica en la lógica de inferencia bayesiana.

-----

## ⚙️ Instalación

Este proyecto utiliza librerías comunes de simulación cuántica y ML.

### Dependencias Principales

```bash
pip install numpy scipy scikit-learn tensorflow tensorflow-probability matplotlib plotly
```

### Estructura de Archivos

Asegúrate de tener los siguientes archivos base en tu directorio:

```
.
├── quantum_error_correction_fixed.py
├── quantum_bayes_mahalanobis.py
├── bayes_logic.py
├── quantum_nn.py 
└── main.py         # Script de orquestación
```

-----

## 🚀 Uso y Ejecución

El script `main.py` es el punto de entrada para probar la funcionalidad integrada.

### 1\. Configuración (Ejemplo)

El `AdvancedQNN` se configura usando el dataclass `QNNConfig`:

```python
from quantum_nn import QNNConfig
"""
    Cada estado en superposición contiene TODA la información:
|ψ⟩ = Σ(αᵢ|CircuitA⟩ + βᵢ|CircuitB⟩) 
     para i = 0 to 1023
"""
config = QNNConfig(
    circuit_A_layers=2,
    circuit_B_layers=2,
    entanglement_layers=1,
    optimizer='adam',
    learning_rate=0.01,
    epochs=50,
    l2_reg=0.001  # Regularización clásica
)
```

### 2\. Entrenamiento y Auditoría

La integración se realiza al añadir el término Mahalanobis a la función de pérdida de la QNN:

$$L_{Total} = L_{MSE} + \lambda \cdot D_{Mahalanobis}(\vec{\theta}, \vec{\theta}_{Prior})$$

Esto asegura que el entrenamiento de la QNN se mantenga **físicamente informado**, penalizando los parámetros que se alejen de un espacio de estados cuánticos deseable.

### 3\. Ejecución

Ejecuta el orquestador principal:

```bash
python main.py
```
### Extensiones Futuras Identificadas

1.  **Modelos de Ruido Avanzados:** Incluir soporte completo para *Amplitude Damping* y *Phase Damping*.
2.  **Integración de Hardware:** Adaptar los circuitos a *backends* reales usando frameworks como Qiskit o Cirq.
3.  **Algoritmos Evolutivos:** Implementar Natural Gradient Descent Cuántico para una optimización más eficiente.
-----
<div align="center">
   <p> 
        
      🤝 Contribuciones y Extensiones
        Agradecemos cualquier contribución que fortalezca el framework.

   </p>
   
</div>
