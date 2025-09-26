import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, load_breast_cancer
import sqlite3
import json
import time
from tqdm import tqdm 

@dataclass
class QNNConfig:
    """Configuración avanzada de la QNN"""
    # Arquitectura de circuitos
    circuit_A_layers: int = 2
    circuit_B_layers: int = 2
    entanglement_layers: int = 1

    # Tipos de puertas por capa
    gate_types_A: List[str] = field(default_factory=lambda: ['RY', 'RZ'])
    gate_types_B: List[str] = field(default_factory=lambda: ['RY', 'RZ'])

    # Patrones de entrelazamiento
    entanglement_pattern: str = 'linear' # 'linear', 'circular', 'all_to_all'

    # Optimización
    optimizer: str = 'adam' # 'sgd', 'adam', 'rmsprop', 'quantum_natural'
    learning_rate: float = 0.01
    epochs: int = 20
    batch_size: int = 8

    # Parámetros del optimizador
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    # Regularización
    l1_reg: float = 0.0
    l2_reg: float = 0.001
    dropout_prob: float = 0.0

    # Técnicas avanzadas
    param_noise: float = 0.01 # Ruido en parámetros para exploración

class DatabaseManager:
    """Gestor de base de datos para entrenamientos QNN"""

    def __init__(self, db_path: str = "qnn_experiments.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, config TEXT NOT NULL,
                    dataset_info TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, status TEXT DEFAULT 'created'
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id INTEGER, epoch INTEGER, loss REAL,
                    accuracy REAL, entanglement_entropy REAL, gradient_norm REAL, param_norm REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id INTEGER, epoch INTEGER, circuit TEXT,
                    layer INTEGER, qubit INTEGER, angle REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            ''')
            conn.commit()

    def save_experiment(self, name: str, config: QNNConfig, dataset_info: Dict) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            config_json = json.dumps(config.__dict__, default=str)

            # Convert numpy integers to Python integers for JSON serialization
            dataset_info_serializable = {k: int(v) if isinstance(v, np.integer) else v for k, v in dataset_info.items()}

            # Ensure values within target_distribution are also converted
            if 'target_distribution' in dataset_info_serializable and isinstance(dataset_info_serializable['target_distribution'], dict):
                 dataset_info_serializable['target_distribution'] = {k: int(v) if isinstance(v, np.integer) else v for k, v in dataset_info_serializable['target_distribution'].items()}


            dataset_json = json.dumps(dataset_info_serializable)

            cursor.execute(
                'INSERT INTO experiments (name, config, dataset_info, status) VALUES (?, ?, ?, ?)',
                (name, config_json, dataset_json, 'running')
            )
            experiment_id = cursor.lastrowid
            conn.commit()
            return experiment_id

    def log_training_step(self, experiment_id: int, epoch: int, metrics: Dict):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_metrics
                (experiment_id, epoch, loss, accuracy, entanglement_entropy, gradient_norm, param_norm)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (experiment_id, epoch, metrics.get('loss', 0), metrics.get('accuracy', 0),
                  metrics.get('entanglement_entropy', 0), metrics.get('grad_norm', 0),
                  metrics.get('param_norm', 0)))
            conn.commit()

    def log_parameters(self, experiment_id: int, epoch: int, qnn_params):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            params_to_log = []
            for layer, angles in enumerate(qnn_params.rotation_angles_A):
                for qubit, angle in enumerate(angles):
                    params_to_log.append((experiment_id, epoch, 'A', layer, qubit, angle))
            for layer, angles in enumerate(qnn_params.rotation_angles_B):
                for qubit, angle in enumerate(angles):
                    params_to_log.append((experiment_id, epoch, 'B', layer, qubit, angle))

            cursor.executemany(
                'INSERT INTO parameters (experiment_id, epoch, circuit, layer, qubit, angle) VALUES (?, ?, ?, ?, ?, ?)',
                params_to_log
            )
            conn.commit()

    def update_experiment_status(self, experiment_id: int, status: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE experiments SET status = ? WHERE id = ?', (status, experiment_id))
            conn.commit()

# --- Clase de Optimizador (Añadido RMSprop) ---

@dataclass
class QNNParams:
    """Dataclass para almacenar todos los parámetros de la QNN."""
    rotation_angles_A: np.ndarray
    rotation_angles_B: np.ndarray
    classical_weights: np.ndarray # Pesos para combinar salidas

class QuantumOptimizer:
    """Optimizadores cuánticos avanzados"""
    def __init__(self, config: QNNConfig):
        self.config = config
        self.reset_state()

    def reset_state(self):
        self.m, self.v = {}, {} # Diccionarios para almacenar estados por tipo de parámetro
        self.t = 0

    def update_parameters(self, params: QNNParams, gradients: QNNParams) -> Dict:
        self.t += 1
        optimizer_map = {
            'sgd': self._sgd_update,
            'adam': self._adam_update,
            'rmsprop': self._rmsprop_update,
        }
        if self.config.optimizer not in optimizer_map:
            raise ValueError(f"Optimizador no reconocido: {self.config.optimizer}")

        return optimizer_map[self.config.optimizer](params, gradients)

    def _initialize_state_for_param(self, name: str, shape: tuple):
        """Inicializa los estados del optimizador para un conjunto de parámetros."""
        if name not in self.m:
            self.m[name] = np.zeros(shape)
            self.v[name] = np.zeros(shape)

    def _sgd_update(self, params: QNNParams, gradients: QNNParams) -> Dict:
        grad_norm = 0
        param_norm = 0

        for name, grad_arr in gradients.__dict__.items():
            param_arr = getattr(params, name)
            self._initialize_state_for_param(name, param_arr.shape)

            # Momentum
            self.m[name] = self.config.momentum * self.m[name] + grad_arr
            update = self.config.learning_rate * self.m[name]

            # Regularización L2
            update += self.config.learning_rate * self.config.l2_reg * param_arr

            param_arr -= update

            # Regularización L1 (soft thresholding)
            if self.config.l1_reg > 0:
                setattr(params, name, np.sign(param_arr) * np.maximum(np.abs(param_arr) - self.config.l1_reg, 0))

            grad_norm += np.sum(grad_arr**2)
            param_norm += np.sum(param_arr**2)

        return {'grad_norm': np.sqrt(grad_norm), 'param_norm': np.sqrt(param_norm)}

    def _adam_update(self, params: QNNParams, gradients: QNNParams) -> Dict:
        grad_norm = 0
        param_norm = 0

        for name, grad_arr in gradients.__dict__.items():
            param_arr = getattr(params, name)
            self._initialize_state_for_param(name, param_arr.shape)

            self.m[name] = self.config.beta1 * self.m[name] + (1 - self.config.beta1) * grad_arr
            self.v[name] = self.config.beta2 * self.v[name] + (1 - self.config.beta2) * (grad_arr**2)

            m_hat = self.m[name] / (1 - self.config.beta1**self.t)
            v_hat = self.v[name] / (1 - self.config.beta2**self.t)

            update = self.config.learning_rate * m_hat / (np.sqrt(v_hat) + self.config.epsilon)
            update += self.config.learning_rate * self.config.l2_reg * param_arr

            param_arr -= update

            # Ruido para exploración
            if self.config.param_noise > 0 and name.startswith('rotation'):
                 param_arr += np.random.normal(0, self.config.param_noise, param_arr.shape)

            grad_norm += np.sum(grad_arr**2)
            param_norm += np.sum(param_arr**2)

        return {'grad_norm': np.sqrt(grad_norm), 'param_norm': np.sqrt(param_norm)}

    def _rmsprop_update(self, params: QNNParams, gradients: QNNParams) -> Dict:
        grad_norm = 0
        param_norm = 0

        for name, grad_arr in gradients.__dict__.items():
            param_arr = getattr(params, name)
            self._initialize_state_for_param(name, param_arr.shape)

            self.v[name] = self.config.beta2 * self.v[name] + (1 - self.config.beta2) * (grad_arr**2)

            update = self.config.learning_rate * grad_arr / (np.sqrt(self.v[name]) + self.config.epsilon)
            update += self.config.learning_rate * self.config.l2_reg * param_arr

            param_arr -= update

            grad_norm += np.sum(grad_arr**2)
            param_norm += np.sum(param_arr**2)

        return {'grad_norm': np.sqrt(grad_norm), 'param_norm': np.sqrt(param_norm)}


# --- Clase QNN Principal (Completada y Optimizada) ---

class AdvancedQNN:
    """QNN Avanzada con todas las optimizaciones técnicas"""

    def __init__(self, config: QNNConfig, num_features: int):
        self.config = config
        self.num_qubits_A = 5
        self.num_qubits_B = 5
        self.total_qubits = self.num_qubits_A + self.num_qubits_B

        if self.total_qubits > 14:
            print(f"Advertencia: Simular {self.total_qubits} qubits es computacionalmente muy costoso.")

        self.dim = 2 ** self.total_qubits
        self.num_features = num_features

        self.params = self._initialize_parameters()
        self.optimizer = QuantumOptimizer(config)
        self.input_scaler = StandardScaler()

        self.db = DatabaseManager()
        self.experiment_id = None

        self.training_history = {
            'loss': [], 'accuracy': [], 'entanglement_entropy': [],
            'grad_norm': [], 'param_norm': []
        }
        self._gate_matrices = {
            'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.array([[1, 0], [0, -1]])
        }

    def _initialize_parameters(self) -> QNNParams:
        # Inicialización Xavier adaptada
        var_A = 2.0 / (self.num_qubits_A * self.config.circuit_A_layers)
        var_B = 2.0 / (self.num_qubits_B * self.config.circuit_B_layers)

        return QNNParams(
            rotation_angles_A=np.random.normal(0, np.sqrt(var_A), (self.config.circuit_A_layers, self.num_qubits_A)),
            rotation_angles_B=np.random.normal(0, np.sqrt(var_B), (self.config.circuit_B_layers, self.num_qubits_B)),
            classical_weights=np.random.randn(self.num_qubits_A + self.num_qubits_B) * 0.1
        )

    def preprocess_data(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.input_scaler.fit(X)
        X_scaled = self.input_scaler.transform(X)
        # Normalizar a [0, pi] para angle encoding
        min_max_scaler = MinMaxScaler(feature_range=(0, np.pi))
        return min_max_scaler.fit_transform(X_scaled)

    # --- Métodos de construcción de circuito (optimizados) ---
    def _apply_single_qubit_gate(self, estado: np.ndarray, qubit: int, gate: np.ndarray) -> np.ndarray:
        """Aplica una puerta de un solo qubit de forma eficiente."""
        op_list = [self._gate_matrices['I']] * self.total_qubits
        op_list[qubit] = gate

        full_op = op_list[0]
        for op in op_list[1:]:
            full_op = np.kron(full_op, op)

        return full_op @ estado

    def _apply_cnot(self, estado: np.ndarray, control: int, target: int) -> np.ndarray:
        """Aplica una puerta CNOT de forma eficiente."""
        nuevo_estado = estado.copy()
        for i in range(self.dim):
            if (i >> control) & 1: # Si el bit de control es 1
                # Invertir el bit objetivo
                mask = 1 << target
                j = i ^ mask
                nuevo_estado[i], nuevo_estado[j] = estado[j], estado[i]
        return nuevo_estado

    def _create_circuit_layer(self, estado: np.ndarray, angles: np.ndarray, gate_types: List[str], qubit_offset: int) -> np.ndarray:
        for qubit_idx, angle in enumerate(angles):
            qubit = qubit_idx + qubit_offset
            for gate_type in gate_types:
                if gate_type == 'RX':
                    gate = np.array([[np.cos(angle/2), -1j*np.sin(angle/2)], [-1j*np.sin(angle/2), np.cos(angle/2)]])
                elif gate_type == 'RY':
                    gate = np.array([[np.cos(angle/2), -np.sin(angle/2)], [np.sin(angle/2), np.cos(angle/2)]])
                elif gate_type == 'RZ':
                    gate = np.array([[np.exp(-1j*angle/2), 0], [0, np.exp(1j*angle/2)]])
                else:
                    continue
                estado = self._apply_single_qubit_gate(estado, qubit, gate)
        return estado

    def _apply_entanglement_pattern(self, estado: np.ndarray, qubits: range, pattern: str) -> np.ndarray:
        qubits_list = list(qubits)
        if pattern == 'linear' and len(qubits_list) > 1:
            for i in range(len(qubits_list) - 1):
                estado = self._apply_cnot(estado, qubits_list[i], qubits_list[i + 1])
        # Otros patrones se pueden implementar aquí de forma similar
        return estado

    def _encode_classical_data(self, input_data: np.ndarray) -> np.ndarray:
        """Codificación de datos usando rotaciones."""
        estado = np.zeros(self.dim, dtype=np.complex128)
        estado[0] = 1.0 # Empezar en |00...0>

        for i in range(self.total_qubits):
            # Rotar cada qubit según una característica de entrada
            feature_idx = i % len(input_data)
            angle = input_data[feature_idx]
            ry_gate = np.array([[np.cos(angle/2), -np.sin(angle/2)], [np.sin(angle/2), np.cos(angle/2)]])
            estado = self._apply_single_qubit_gate(estado, i, ry_gate)
        return estado

    # --- Forward Pass y Medición ---

    def forward_pass(self, input_data: np.ndarray) -> Tuple[float, np.ndarray]:
        estado = self._encode_classical_data(input_data)

        # Circuito A
        for layer in range(self.config.circuit_A_layers):
            estado = self._create_circuit_layer(estado, self.params.rotation_angles_A[layer], self.config.gate_types_A, 0)
            estado = self._apply_entanglement_pattern(estado, range(self.num_qubits_A), self.config.entanglement_pattern)

        # Circuito B
        for layer in range(self.config.circuit_B_layers):
            estado = self._create_circuit_layer(estado, self.params.rotation_angles_B[layer], self.config.gate_types_B, self.num_qubits_A)
            estado = self._apply_entanglement_pattern(estado, range(self.num_qubits_A, self.total_qubits), self.config.entanglement_pattern)

        # Entrelazamiento inter-circuitos
        for _ in range(self.config.entanglement_layers):
            for i in range(self.num_qubits_A):
                estado = self._apply_cnot(estado, i, i + self.num_qubits_A)

        # Medición
        measurements = np.array([self._measure_qubit(estado, i) for i in range(self.total_qubits)])

        # Combinación clásica
        output = np.tanh(np.dot(self.params.classical_weights, measurements)) # Usar tanh para acotar la salida a [-1, 1]

        return output, estado

    def _measure_qubit(self, estado: np.ndarray, qubit: int) -> float:
        """Calcula el valor esperado del observable Z en un qubit."""
        op_z_list = [self._gate_matrices['I']] * self.total_qubits
        op_z_list[qubit] = self._gate_matrices['Z']

        observable = op_z_list[0]
        for op in op_z_list[1:]:
            observable = np.kron(observable, op)

        return np.real(estado.conj().T @ observable @ estado)

    # --- Cálculo de Gradientes y Pérdida ---

    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean((y_pred - y_true)**2) # Mean Squared Error

    def parameter_shift_gradient(self, input_data: np.ndarray, y_true: float) -> QNNParams:
        shift = np.pi / 2
        original_params = self.params

        # Clonar los parámetros para no modificarlos directamente
        params_copy = QNNParams(
            rotation_angles_A=original_params.rotation_angles_A.copy(),
            rotation_angles_B=original_params.rotation_angles_B.copy(),
            classical_weights=original_params.classical_weights.copy()
        )
        self.params = params_copy

        grads = QNNParams(
            rotation_angles_A=np.zeros_like(self.params.rotation_angles_A),
            rotation_angles_B=np.zeros_like(self.params.rotation_angles_B),
            classical_weights=np.zeros_like(self.params.classical_weights)
        )

        param_sets = [
            (grads.rotation_angles_A, self.params.rotation_angles_A),
            (grads.rotation_angles_B, self.params.rotation_angles_B),
            (grads.classical_weights, self.params.classical_weights)
        ]

        for grad_arr, param_arr in param_sets:
            it = np.nditer(param_arr, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                original_val = param_arr[idx]

                param_arr[idx] = original_val + shift
                pred_plus, _ = self.forward_pass(input_data)

                param_arr[idx] = original_val - shift
                pred_minus, _ = self.forward_pass(input_data)

                # Gradiente de la pérdida MSE: 2 * (y_pred - y_true) * d(y_pred)/d(theta)
                # d(loss)/d(theta) = (loss(+) - loss(-)) / 2  no es correcto para MSE
                # d(loss)/d(theta) = d(loss)/d(pred) * d(pred)/d(theta)
                y_pred_original, _ = self.forward_pass(input_data)
                grad_loss_pred = 2 * (y_pred_original - y_true)

                grad_pred_theta = (pred_plus - pred_minus) / 2.0
                grad_arr[idx] = grad_loss_pred * grad_pred_theta

                param_arr[idx] = original_val
                it.iternext()

        self.params = original_params # Restaurar
        return grads

    # --- Lógica de Entrenamiento y Evaluación ---

    def fit(self, X: np.ndarray, y: np.ndarray, experiment_name: str):
        X_proc = self.preprocess_data(X, fit=True)
        # Mapear y a [-1, 1] para coincidir con la salida de tanh
        y_proc = 2 * y - 1

        dataset_info = {'n_samples': len(X), 'n_features': X.shape[1], 'target_distribution': dict(pd.Series(y).value_counts())}
        self.experiment_id = self.db.save_experiment(experiment_name, self.config, dataset_info)

        n_batches = len(X) // self.config.batch_size

        for epoch in range(self.config.epochs):
            epoch_loss = 0

            with tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{self.config.epochs}") as pbar:
                for i in pbar:
                    start = i * self.config.batch_size
                    end = start + self.config.batch_size
                    X_batch, y_batch = X_proc[start:end], y_proc[start:end]

                    batch_grads = QNNParams(
                        rotation_angles_A=np.zeros_like(self.params.rotation_angles_A),
                        rotation_angles_B=np.zeros_like(self.params.rotation_angles_B),
                        classical_weights=np.zeros_like(self.params.classical_weights)
                    )

                    for x_sample, y_sample in zip(X_batch, y_batch):
                        grads = self.parameter_shift_gradient(x_sample, y_sample)
                        batch_grads.rotation_angles_A += grads.rotation_angles_A
                        batch_grads.rotation_angles_B += grads.rotation_angles_B
                        batch_grads.classical_weights += grads.classical_weights

                    # Promediar gradientes del batch
                    batch_grads.rotation_angles_A /= self.config.batch_size
                    batch_grads.rotation_angles_B /= self.config.batch_size
                    batch_grads.classical_weights /= self.config.batch_size

                    # Actualizar parámetros
                    opt_metrics = self.optimizer.update_parameters(self.params, batch_grads)

            # Métricas al final de la época
            preds, _ = self.predict(X, raw_output=True)
            loss = self._compute_loss(preds, y_proc)
            accuracy = self.evaluate(X, y)['accuracy']
            ent_entropy = self._compute_entanglement_entropy(X_proc[0]) # En una muestra

            metrics = {
                'loss': loss, 'accuracy': accuracy, 'entanglement_entropy': ent_entropy,
                **opt_metrics
            }

            self.training_history['loss'].append(loss)
            self.training_history['accuracy'].append(accuracy)
            self.training_history['entanglement_entropy'].append(ent_entropy)
            self.training_history['grad_norm'].append(opt_metrics.get('grad_norm'))
            self.training_history['param_norm'].append(opt_metrics.get('param_norm'))

            self.db.log_training_step(self.experiment_id, epoch, metrics)
            self.db.log_parameters(self.experiment_id, epoch, self.params)

            print(f"Epoch {epoch+1} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Entanglement: {ent_entropy:.4f}")

        self.db.update_experiment_status(self.experiment_id, 'completed')
        print("Entrenamiento completado.")

    def predict(self, X: np.ndarray, raw_output: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        X_proc = self.preprocess_data(X)
        predictions = np.array([self.forward_pass(x)[0] for x in X_proc])
        if raw_output:
            return predictions, (predictions > 0).astype(int)
        return (predictions > 0).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        _, y_pred_class = self.predict(X, raw_output=True)
        accuracy = np.mean(y_pred_class == y)
        return {'accuracy': accuracy}

    # --- Métodos de Análisis y Visualización ---

    def _compute_entanglement_entropy(self, input_data: np.ndarray) -> float:
        """Calcula la entropía de entrelazamiento entre el subsistema A y B."""
        _, final_state = self.forward_pass(input_data)

        # Trazar sobre el subsistema B para obtener la matriz de densidad reducida de A
        rho = np.outer(final_state, final_state.conj())
        rho_A = np.trace(rho.reshape(2**self.num_qubits_A, 2**self.num_qubits_B,
                                     2**self.num_qubits_A, 2**self.num_qubits_B), axis1=1, axis2=3)

        # Calcular autovalores y entropía de Von Neumann
        eigenvalues = np.linalg.eigvalsh(rho_A)
        non_zero_eigvals = eigenvalues[eigenvalues > 1e-12] # Evitar log(0)
        entropy = -np.sum(non_zero_eigvals * np.log2(non_zero_eigvals))
        return np.real(entropy)

    def visualize_training(self):
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Resultados del Entrenamiento (Experimento {self.experiment_id})")

        axs[0, 0].plot(self.training_history['loss'], 'r-o', label='Loss')
        axs[0, 0].set_title('Pérdida vs. Época')
        axs[0, 0].set_xlabel('Época')
        axs[0, 0].set_ylabel('Pérdida (MSE)')
        axs[0, 0].grid(True)

        axs[0, 1].plot(self.training_history['accuracy'], 'b-o', label='Accuracy')
        axs[0, 1].set_title('Precisión vs. Época')
        axs[0, 1].set_xlabel('Época')
        axs[0, 1].set_ylabel('Precisión')
        axs[0, 1].grid(True)

        axs[1, 0].plot(self.training_history['entanglement_entropy'], 'g-o', label='Entanglement')
        axs[1, 0].set_title('Entropía de Entrelazamiento vs. Época')
        axs[1, 0].set_xlabel('Época')
        axs[1, 0].set_ylabel('Entropía (bits)')
        axs[1, 0].grid(True)

        axs[1, 1].plot(self.training_history['grad_norm'], 'm-o', label='Norma del Gradiente')
        axs[1, 1].plot(self.training_history['param_norm'], 'c-o', label='Norma de Parámetros')
        axs[1, 1].set_title('Normas vs. Época')
        axs[1, 1].set_xlabel('Época')
        axs[1, 1].set_ylabel('Valor de la Norma')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

# --- Ejemplo de Uso ---

if __name__ == "__main__":
    # 1. Cargar y preparar datos
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Reducir el número de características para que coincida con los qubits de codificación
    num_features_to_use = 10
    X = X[:, :num_features_to_use]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 2. Configurar la QNN
    qnn_config = QNNConfig(
        circuit_A_layers=2,
        circuit_B_layers=2,
        optimizer='adam',
        learning_rate=0.05,
        epochs=10, # Bajas para una prueba rápida
        batch_size=16
    )

    # 3. Crear y entrenar el modelo
    print("Inicializando la QNN Avanzada...")
    qnn = AdvancedQNN(config=qnn_config, num_features=X_train.shape[1])

    print("Iniciando entrenamiento...")
    start_time = time.time()
    qnn.fit(X_train, y_train, experiment_name="BreastCancer_Adam_Test")
    end_time = time.time()
    print(f"Tiempo de entrenamiento: {end_time - start_time:.2f} segundos")

    # 4. Evaluar el modelo
    print("\nEvaluando en el conjunto de prueba...")
    test_metrics = qnn.evaluate(X_test, y_test)
    print(f"Precisión en prueba: {test_metrics['accuracy']:.4f}")

    # 5. Visualizar resultados
    qnn.visualize_training()

    # 6. Consultar la base de datos (ejemplo)
    print("\nConsultando la base de datos para el último experimento...")
    try:
        conn = sqlite3.connect("qnn_experiments.db")
        df_metrics = pd.read_sql_query(f"SELECT * FROM training_metrics WHERE experiment_id = {qnn.experiment_id}", conn)
        print("Métricas del experimento:")
        print(df_metrics.head())
        conn.close()
    except Exception as e:
        print(f"No se pudo consultar la base de datos: {e}")
