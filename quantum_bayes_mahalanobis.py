# quantum_bayes_mahalanobis.py

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance

# Ajusta el import a tus necesidades reales:
# Se asume que en bayes_logic.py se tienen las clases BayesLogic, PRN, además de las
# funciones shannon_entropy, calculate_cosines, etc.
from protocolo.bayes_logic import (
    BayesLogic,
    PRN,
    shannon_entropy,
    calculate_cosines  
)

class QuantumBayesMahalanobis(BayesLogic):
    """
    Clase que combina la lógica de Bayes con el cálculo de la distancia de Mahalanobis
    aplicada a estados cuánticos, permitiendo proyecciones vectorizadas e inferencias
    de coherencia/entropía.
    """
    def __init__(self):
        """
        Constructor que inicializa el estimador de covarianza para su posterior uso.
        """
        super().__init__()
        self.covariance_estimator = EmpiricalCovariance()

    def _get_inverse_covariance(self, data: np.ndarray) -> np.ndarray:
        """
        Ajusta el estimador de covarianza con los datos y retorna la inversa de la
        matriz de covarianza. Si la matriz no fuera invertible, se retorna la
        pseudo-inversa (pinv).

        Parámetros:
        -----------
        data: np.ndarray
            Datos con forma (n_muestras, n_dimensiones).

        Retorna:
        --------
        inv_cov_matrix: np.ndarray
            Inversa o pseudo-inversa de la matriz de covarianza estimada.
        """
        if data.ndim != 2:
            raise ValueError("Los datos deben ser una matriz bidimensional (n_muestras, n_dimensiones).")
        self.covariance_estimator.fit(data)
        cov_matrix = self.covariance_estimator.covariance_
        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            inv_cov_matrix = np.linalg.pinv(cov_matrix)
        return inv_cov_matrix

    def compute_quantum_mahalanobis(self,
                                    quantum_states_A: np.ndarray,
                                    quantum_states_B: np.ndarray) -> np.ndarray:
        """
        Calcula la distancia de Mahalanobis para cada estado en 'quantum_states_B'
        respecto a la distribución de 'quantum_states_A'. Retorna un arreglo 1D
        con tantas distancias como filas tenga 'quantum_states_B'.

        Parámetros:
        -----------
        quantum_states_A: np.ndarray
            Representa el conjunto de estados cuánticos de referencia.
            Forma esperada: (n_muestras, n_dimensiones).

        quantum_states_B: np.ndarray
            Estados cuánticos para los que calcularemos la distancia
            de Mahalanobis. Forma: (n_muestras, n_dimensiones).

        Retorna:
        --------
        distances: np.ndarray
            Distancias de Mahalanobis calculadas para cada entrada de B.
        """
        if quantum_states_A.ndim != 2 or quantum_states_B.ndim != 2:
            raise ValueError("Los estados cuánticos deben ser matrices bidimensionales.")
        if quantum_states_A.shape[1] != quantum_states_B.shape[1]:
            raise ValueError("La dimensión (n_dimensiones) de A y B debe coincidir.")

        inv_cov_matrix = self._get_inverse_covariance(quantum_states_A)
        mean_A = np.mean(quantum_states_A, axis=0)

        diff_B = quantum_states_B - mean_A  # (n_samples_B, n_dims)
        aux = diff_B @ inv_cov_matrix       # (n_samples_B, n_dims)
        dist_sqr = np.einsum('ij,ij->i', aux, diff_B)  # Producto elemento a elemento y sumatoria por fila
        distances = np.sqrt(dist_sqr)
        return distances

    def quantum_cosine_projection(self,
                                  quantum_states: np.ndarray,
                                  entropy: float,
                                  coherence: float) -> tf.Tensor:
        """
        Proyecta los estados cuánticos usando cosenos directores y calcula la
        distancia de Mahalanobis entre dos proyecciones vectorizadas (A y B).
        Finalmente retorna las distancias normalizadas (softmax).

        Parámetros:
        -----------
        quantum_states: np.ndarray
            Estados cuánticos de entrada con forma (n_muestras, 2).
        entropy: float
            Entropía del sistema a usar en la función calculate_cosines.
        coherence: float
            Coherencia del sistema a usar en la función calculate_cosines.

        Retorna:
        --------
        normalized_distances: tf.Tensor
            Tensor 1D con las distancias normalizadas (softmax).
        """
        if quantum_states.shape[1] != 2:
            raise ValueError("Se espera que 'quantum_states' tenga exactamente 2 columnas.")
        cos_x, cos_y, cos_z = calculate_cosines(entropy, coherence)

        # Proyección A: multiplicar cada columna por (cos_x, cos_y)
        projected_states_A = quantum_states * np.array([cos_x, cos_y])
        # Proyección B: multiplicar cada columna por (cos_x*cos_z, cos_y*cos_z)
        projected_states_B = quantum_states * np.array([cos_x * cos_z, cos_y * cos_z])

        # Calcular distancias de Mahalanobis vectorizadas
        mahalanobis_distances = self.compute_quantum_mahalanobis(
            projected_states_A,
            projected_states_B
        )

        # Convertir a tensor y normalizar con softmax
        mahalanobis_distances_tf = tf.convert_to_tensor(mahalanobis_distances, dtype=tf.float32)
        normalized_distances = tf.nn.softmax(mahalanobis_distances_tf)
        return normalized_distances

    def calculate_quantum_posterior_with_mahalanobis(self,
                                                     quantum_states: np.ndarray,
                                                     entropy: float,
                                                     coherence: float):
        """
        Calcula la probabilidad posterior usando la distancia de Mahalanobis
        en proyecciones cuánticas e integra la lógica de Bayes.

        Parámetros:
        -----------
        quantum_states: np.ndarray
            Matriz de estados cuánticos (n_muestras, 2).
        entropy: float
            Entropía del sistema.
        coherence: float
            Coherencia del sistema.

        Retorna:
        --------
        posterior: tf.Tensor
            Probabilidad posterior calculada combinando la lógica bayesiana.
        quantum_projections: tf.Tensor
            Proyecciones cuánticas normalizadas (distancias softmax).
        """
        quantum_projections = self.quantum_cosine_projection(
            quantum_states,
            entropy,
            coherence
        )

        # Calcular covarianza en las proyecciones
        tensor_projections = tf.convert_to_tensor(quantum_projections, dtype=tf.float32)
        quantum_covariance = tfp.stats.covariance(tensor_projections, sample_axis=0)

        # Calcular prior cuántico basado en la traza de la covarianza
        dim = tf.cast(tf.shape(quantum_covariance)[0], tf.float32)
        quantum_prior = tf.linalg.trace(quantum_covariance) / dim

        # Calcular otros componentes para la posteriori (usando métodos heredados de BayesLogic).
        prior_coherence = self.calculate_high_coherence_prior(coherence)
        joint_prob = self.calculate_joint_probability(
            coherence,
            1,  # variable arbitraria: "evento" = 1
            tf.reduce_mean(tensor_projections)
        )
        cond_prob = self.calculate_conditional_probability(joint_prob, quantum_prior)
        posterior = self.calculate_posterior_probability(quantum_prior,
                                                         prior_coherence,
                                                         cond_prob)
        return posterior, quantum_projections

    def predict_quantum_state(self,
                              quantum_states: np.ndarray,
                              entropy: float,
                              coherence: float):
        """
        Predice el siguiente estado cuántico con base en la proyección y la distancia
        de Mahalanobis, generando un "estado futuro".

        Parámetros:
        -----------
        quantum_states: np.ndarray
            Estados cuánticos de entrada (n_muestras, 2).
        entropy: float
            Entropía del sistema.
        coherence: float
            Coherencia del sistema.

        Retorna:
        --------
        next_state_prediction: tf.Tensor
            Predicción del siguiente estado cuántico.
        posterior: tf.Tensor
            Probabilidad posterior que se usó en la predicción.
        """
        posterior, projections = self.calculate_quantum_posterior_with_mahalanobis(
            quantum_states,
            entropy,
            coherence
        )

        # Generar un estado futuro ponderado por la posterior
        # Posterior es escalar, mientras que projections es un vector
        next_state_prediction = tf.reduce_sum(
            tf.multiply(projections, tf.expand_dims(posterior, -1)),
            axis=0
        )
        return next_state_prediction, posterior


class EnhancedPRN(PRN):
    """
    Extiende la clase PRN para registrar distancias de Mahalanobis y con ello
    definir un 'ruido cuántico' adicional en el sistema.
    """
    def __init__(self, influence: float = 0.5, algorithm_type: str = None, **parameters):
        """
        Constructor que permite definir la influencia y el tipo de algoritmo,
        además de inicializar una lista para conservar registros promedio de
        distancias de Mahalanobis.
        """
        super().__init__(influence, algorithm_type, **parameters)
        self.mahalanobis_records = []

    def record_quantum_noise(self, probabilities: dict, quantum_states: np.ndarray):
        """
        Registra un 'ruido cuántico' basado en la distancia de Mahalanobis
        calculada para los estados cuánticos.

        Parámetros:
        -----------
        probabilities: dict
            Diccionario de probabilidades (ej. {"0": p_0, "1": p_1, ...}).
        quantum_states: np.ndarray
            Estados cuánticos (n_muestras, n_dimensiones).

        Retorna:
        --------
        (entropy, mahal_mean): Tuple[float, float]
            - Entropía calculada a partir de probabilities.
            - Distancia promedio de Mahalanobis.
        """
        # Calculamos la entropía (este método se asume en la clase base PRN o BayesLogic).
        entropy = self.record_noise(probabilities)

        # Ajuste del estimador de covarianza
        cov_estimator = EmpiricalCovariance().fit(quantum_states)
        mean_state = np.mean(quantum_states, axis=0)
        inv_cov = np.linalg.pinv(cov_estimator.covariance_)

        # Cálculo vectorizado de la distancia
        diff = quantum_states - mean_state
        aux = diff @ inv_cov
        dist_sqr = np.einsum('ij,ij->i', aux, diff)
        distances = np.sqrt(dist_sqr)
        mahal_mean = np.mean(distances)
        
def von_neumann_entropy(density_matrix: np.ndarray) -> float:
    """Calcula la entropía de von Neumann para una matriz de densidad."""
    # Se calculan los valores propios (eigenvalues)
    eigenvalues = np.linalg.eigvalsh(density_matrix)
    # Se filtran los valores propios que son cero o negativos para evitar errores en el logaritmo
    non_zero_eigenvalues = eigenvalues[eigenvalues > 0]
    # Se calcula la entropía: S = -Tr(ρ log(ρ)) = -Σ λ_i log(λ_i)
    entropy = -np.sum(non_zero_eigenvalues * np.log(non_zero_eigenvalues))
    # Se registra la distancia promedio
        self.mahalanobis_records.append(mahal_mean)

        return entropy, mahal_mean

class QuantumNoiseCollapse(QuantumBayesMahalanobis):
    """
    Combina la lógica bayesiana cuántica (QuantumBayesMahalanobis) y el registro ExtendedPRN
    para simular el 'colapso de onda' usando distancias de Mahalanobis como parte del ruido.
    """
    def __init__(self, prn_influence: float = 0.5):
        """
        Constructor que crea internamente un EnhancedPRN por defecto, con una
        influencia configurable.
        """
        super().__init__()
        self.prn = EnhancedPRN(influence=prn_influence)

    def simulate_wave_collapse(self,
                               quantum_states: np.ndarray,
                               prn_influence: float,
                               previous_action: int):
        """
        Simula el colapso de onda incorporando ruido cuántico (a través de PRN) e
        integra el resultado para determinar una acción bayesiana.

        Parámetros:
        -----------
        quantum_states: np.ndarray
            Estados cuánticos de entrada.
        prn_influence: float
            Influencia del PRN en el sistema (se puede alinear con self.prn.influence).
        previous_action: int
            Acción previa del sistema que se utiliza como condicionante.

        Retorna:
        --------
        dict con llaves:
            "collapsed_state": tf.Tensor
                Representación final colapsada del estado.
            "action": int
                Acción tomada según lógica bayesiana.
            "entropy": float
                Entropía calculada.
            "coherence": float
                Coherencia derivada.
            "mahalanobis_distance": float
                Distancia promedio de Mahalanobis.
            "cosines": Tuple[float, float, float]
                Valores de (cos_x, cos_y, cos_z) usados en la proyección.
        """
        # Diccionario de probabilidades a modo de ejemplo
        probabilities = {str(i): np.sum(state) for i, state in enumerate(quantum_states)}

        # Registro de entropía y distancia de Mahalanobis
        entropy, mahalanobis_mean = self.prn.record_quantum_noise(probabilities, quantum_states)

        # Cálculo de los cosenos directores como ejemplo de proyección
        cos_x, cos_y, cos_z = calculate_cosines(entropy, mahalanobis_mean)

        # Definimos coherencia a partir de la distancia de Mahalanobis y los cosenos
        coherence = np.exp(-mahalanobis_mean) * (cos_x + cos_y + cos_z) / 3.0

        # Llamada a un método de BayesLogic para decidir la acción
        bayes_probs = self.calculate_probabilities_and_select_action(
            entropy=entropy,
            coherence=coherence,
            prn_influence=prn_influence,
            action=previous_action
        )

        # Proyectar estados cuánticos
        projected_states = self.quantum_cosine_projection(
            quantum_states,
            entropy,
            coherence
        )

        # Ejemplo de 'colapso' multiplicando la proyección por la acción que se toma
        collapsed_state = tf.reduce_sum(
            tf.multiply(
                projected_states,
                tf.cast(bayes_probs["action_to_take"], tf.float32)
            )
        )

        return {
            "collapsed_state": collapsed_state,
            "action": bayes_probs["action_to_take"],
            "entropy": entropy,
            "coherence": coherence,
            "mahalanobis_distance": mahalanobis_mean,
            "cosines": (cos_x, cos_y, cos_z)
        }

    def objective_function_with_noise(self,
                                      quantum_states: np.ndarray,
                                      target_state: np.ndarray,
                                      entropy_weight: float = 1.0) -> tf.Tensor:
        """
        Función objetivo que combina fidelidad, entropía y distancia de Mahalanobis
        para encontrar un compromiso entre mantener la fidelidad al estado objetivo
        y el ruido cuántico en el sistema.

        Parámetros:
        -----------
        quantum_states: np.ndarray
            Estados cuánticos actuales (n_muestras, n_dimensiones).
        target_state: np.ndarray
            Estado objetivo que se desea alcanzar.
        entropy_weight: float
            Factor que pondera la influencia de la entropía en la función objetivo.

        Retorna:
        --------
        objective_value: tf.Tensor
            Valor de la función objetivo (cuanto menor, mejor).
        """
        # Calcular fidelidad (simple ejemplo): |<ψ|φ>|^2
        # Suponiendo que (quantum_states y target_state) sean vectores compatibles
        fidelity = tf.abs(tf.reduce_sum(quantum_states * tf.cast(target_state, quantum_states.dtype)))**2

        # Registrar 'ruido': entropía y distancia de Mahalanobis
        probabilities = {str(i): np.sum(st) for i, st in enumerate(quantum_states)}
        entropy, mahalanobis_dist = self.prn.record_quantum_noise(probabilities, quantum_states)

        # Combinar métricas: (1 - fidelidad) + factor * entropía + penalización por distancia
        objective_value = ((1.0 - fidelity)
                           + entropy_weight * entropy
                           + (1.0 - np.exp(-mahalanobis_dist)))

        return objective_value

    def optimize_quantum_state(self,
                               initial_states: np.ndarray,
                               target_state: np.ndarray,
                               max_iterations: int = 100,
                               learning_rate: float = 0.01):
        """
        Optimiza los estados cuánticos para acercarlos al estado objetivo,
        mediante un descenso de gradiente (Adam).

        Parámetros:
        -----------
        initial_states: np.ndarray
            Estados cuánticos iniciales.
        target_state: np.ndarray
            Estado objetivo.
        max_iterations: int
            Número máximo de iteraciones de optimización.
        learning_rate: float
            Tasa de aprendizaje para Adam.

        Retorna:
        --------
        best_states: np.ndarray
            Estados optimizados que reportan el menor valor de la función objetivo.
        best_objective: float
            Valor final alcanzado por la función objetivo.
        """
        # Convertir a tf.Variable para permitir gradientes
        current_states = tf.Variable(initial_states, dtype=tf.float32)

        best_objective = float('inf')
        best_states = current_states.numpy().copy()

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        for _ in range(max_iterations):
            with tf.GradientTape() as tape:
                # Usar numpy() en la llamada para separar lógicamente la parte TF de la parte numpy
                objective = self.objective_function_with_noise(current_states.numpy(), target_state)
            grads = tape.gradient(objective, [current_states])

            if grads[0] is None:
                # Si no hay gradiente, rompe el bucle
                break

            optimizer.apply_gradients(zip(grads, [current_states]))

            # Re-evaluar después de actualizar los parámetros
            new_objective = self.objective_function_with_noise(current_states.numpy(), target_state)
            if new_objective < best_objective:
                best_objective = new_objective
                best_states = current_states.numpy().copy()

        return best_states, best_objective


# ====================
# Ejemplo de uso
# ====================
if __name__ == "__main__":
    qnc = QuantumNoiseCollapse()

    # Estados cuánticos iniciales
    initial_states = np.array([
        [0.8, 0.2],
        [0.9, 0.4],
        [0.1, 0.7]
    ])

    # Estado objetivo
    target_state = np.array([1.0, 0.0])

    # Optimizar estados
    optimized_states, final_objective = qnc.optimize_quantum_state(
        initial_states,
        target_state,
        max_iterations=100,
        learning_rate=0.01
    )

    # Simular colapso final con la acción previa (ej. 0)
    final_collapse = qnc.simulate_wave_collapse(
        optimized_states,
        prn_influence=0.5,
        previous_action=0
    )

    print("Estados optimizados:", optimized_states)
    print("Valor final de la función objetivo:", final_objective)
    print("Resultado del colapso final:", final_collapse)

"""
-------------------------------------------------------------------------------
Notas de la versión refinada:

1) Se agregaron validaciones de dimensión para evitar errores silenciosos.  
2) Se incorporó manejo de excepción cuando la matriz de covarianza no es invertible, 
   recurriendo a la pseudo-inversa.  
3) Se incluyeron docstrings en español más descriptivos para cada clase y método.  
4) Se agregó un argumento learning_rate al método optimize_quantum_state para mayor flexibilidad.  
5) Se salió del bloque de gradient y se volvió a evaluar la función objetivo 
   para comparar si el valor mejoró y así actualizar el mejor estado.  
6) El registro de la distancia de Mahalanobis se realiza cuando sea necesario (e.g. en la función objetivo), 
   evitando duplicar demasiadas llamadas si no se requiere.  
7) El método simulate_wave_collapse mantiene la estructura original pero con más comentarios y 
   simplificaciones de nombres de variables.  
8) Se agrega una funcion de entropia que incluye el metodo de Von Neumann para una dimendion densa cuanticamente correcta
"""
