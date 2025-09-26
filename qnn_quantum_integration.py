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

# Importar clases base de la QNN original
from qnn_advanced import AdvancedQNN, QNNConfig, QNNParams, QuantumOptimizer, DatabaseManager

# Nuevas configuraciones para componentes cuánticos personalizados
@dataclass
class CustomQuantumConfig(QNNConfig):
    """Configuración extendida con componentes cuánticos personalizados"""
    # Parámetros para puertas asimétricas
    asymmetry_factor: float = 0.05
    coupling_strength: float = 0.1
    
    # Patrones de rotación personalizados
    custom_rotation_pattern: str = 'symmetric'  # 'symmetric', 'asymmetric', 'adaptive'
    
    # Análisis de asimetría
    enable_asymmetry_analysis: bool = True
    asymmetry_threshold: float = 0.1

class QuantumAsymmetryAnalyzer:
    """Analizador de asimetría cuántica integrado con QNN"""
    
    def __init__(self):
        self.asymmetry_history = []
        self.coupling_patterns = []
    
    def calculate_state_asymmetry(self, quantum_state: np.ndarray) -> float:
        """
        Calcula la asimetría del estado cuántico
        Basado en la diferencia de probabilidades entre estados |0⟩ y |1⟩
        """
        if len(quantum_state) == 0:
            return 0.0
            
        probabilities = np.abs(quantum_state) ** 2
        
        # Separar estados por paridad de bits
        n_qubits = int(np.log2(len(quantum_state)))
        even_prob = 0.0  # Estados con número par de 1s
        odd_prob = 0.0   # Estados con número impar de 1s
        
        for i, prob in enumerate(probabilities):
            bit_count = bin(i).count('1')
            if bit_count % 2 == 0:
                even_prob += prob
            else:
                odd_prob += prob
        
        # Asimetría como diferencia absoluta
        asymmetry = abs(even_prob - odd_prob)
        return asymmetry
    
    def analyze_coupling_patterns(self, measurements: np.ndarray) -> Dict[str, float]:
        """
        Analiza patrones de acoplamiento en las mediciones
        """
        if len(measurements) == 0:
            return {'coupling_strength': 0.0, 'correlation': 0.0}
            
        # Calcular correlaciones entre qubits adyacentes
        correlations = []
        for i in range(len(measurements) - 1):
            corr = np.corrcoef(measurements[i], measurements[i+1])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        
        coupling_strength = np.mean(correlations) if correlations else 0.0
        
        return {
            'coupling_strength': coupling_strength,
            'correlation': np.mean(correlations) if correlations else 0.0,
            'max_correlation': np.max(correlations) if correlations else 0.0
        }

class EnhancedQNN(AdvancedQNN):
    """QNN mejorada con componentes cuánticos personalizados"""
    
    def __init__(self, config: CustomQuantumConfig, num_features: int):
        super().__init__(config, num_features)
        self.config = config
        self.asymmetry_analyzer = QuantumAsymmetryAnalyzer()
        self.quantum_states_history = []
        
    def _apply_asymmetric_rotation(self, estado: np.ndarray, qubit: int, angle: float, asymmetry_factor: float) -> np.ndarray:
        """
        Aplica rotación asimétrica basada en el patrón personalizado
        """
        if self.config.custom_rotation_pattern == 'asymmetric':
            # Modificar el ángulo con factor de asimetría
            asymmetric_angle = angle * (1 + asymmetry_factor * np.sin(qubit * np.pi / self.total_qubits))
        elif self.config.custom_rotation_pattern == 'adaptive':
            # Ángulo adaptativo basado en el estado actual
            state_norm = np.linalg.norm(estado)
            asymmetric_angle = angle * (1 + asymmetry_factor * state_norm)
        else:
            # Patrón simétrico (default)
            asymmetric_angle = angle
            
        # Crear puerta RY con ángulo modificado
        ry_gate = np.array([[np.cos(asymmetric_angle/2), -np.sin(asymmetric_angle/2)], 
                           [np.sin(asymmetric_angle/2), np.cos(asymmetric_angle/2)]])
        
        return self._apply_single_qubit_gate(estado, qubit, ry_gate)
    
    def _apply_custom_coupling(self, estado: np.ndarray, control: int, target: int, coupling_strength: float) -> np.ndarray:
        """
        Aplica acoplamiento cuántico personalizado entre qubits
        """
        # Aplicar CNOT estándar
        estado = self._apply_cnot(estado, control, target)
        
        # Aplicar rotación controlada adicional basada en coupling_strength
        coupling_angle = coupling_strength * np.pi / 4
        
        # Crear una rotación controlada RZ personalizada
        for i in range(self.dim):
            if (i >> control) & 1:  # Si el bit de control es 1
                # Aplicar rotación de fase al bit objetivo
                if (i >> target) & 1:  # Si el bit objetivo también es 1
                    estado[i] *= np.exp(1j * coupling_angle)
                else:  # Si el bit objetivo es 0
                    estado[i] *= np.exp(-1j * coupling_angle)
        
        return estado
    
    def _create_enhanced_circuit_layer(self, estado: np.ndarray, angles: np.ndarray, 
                                     gate_types: List[str], qubit_offset: int) -> np.ndarray:
        """
        Crea capa de circuito mejorada con componentes personalizados
        """
        for qubit_idx, angle in enumerate(angles):
            qubit = qubit_idx + qubit_offset
            
            for gate_type in gate_types:
                if gate_type == 'RY_ASYM':
                    # Rotación RY asimétrica personalizada
                    estado = self._apply_asymmetric_rotation(
                        estado, qubit, angle, self.config.asymmetry_factor
                    )
                elif gate_type == 'RY':
                    gate = np.array([[np.cos(angle/2), -np.sin(angle/2)], 
                                   [np.sin(angle/2), np.cos(angle/2)]])
                    estado = self._apply_single_qubit_gate(estado, qubit, gate)
                elif gate_type == 'RZ':
                    gate = np.array([[np.exp(-1j*angle/2), 0], 
                                   [0, np.exp(1j*angle/2)]])
                    estado = self._apply_single_qubit_gate(estado, qubit, gate)
                elif gate_type == 'RX':
                    gate = np.array([[np.cos(angle/2), -1j*np.sin(angle/2)], 
                                   [-1j*np.sin(angle/2), np.cos(angle/2)]])
                    estado = self._apply_single_qubit_gate(estado, qubit, gate)
        
        return estado
    
    def _apply_enhanced_entanglement(self, estado: np.ndarray, qubits: range) -> np.ndarray:
        """
        Aplica entrelazamiento mejorado con acoplamiento personalizado
        """
        qubits_list = list(qubits)
        
        if len(qubits_list) > 1:
            for i in range(len(qubits_list) - 1):
                # Aplicar acoplamiento personalizado
                estado = self._apply_custom_coupling(
                    estado, qubits_list[i], qubits_list[i + 1], 
                    self.config.coupling_strength
                )
        
        return estado
    
    def enhanced_forward_pass(self, input_data: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Forward pass mejorado con análisis de asimetría
        """
        estado = self._encode_classical_data(input_data)
        
        # Guardar estado inicial para análisis
        initial_asymmetry = self.asymmetry_analyzer.calculate_state_asymmetry(estado)
        
        # Circuito A con componentes personalizados
        for layer in range(self.config.circuit_A_layers):
            if 'RY_ASYM' in self.config.gate_types_A:
                # Usar puertas asimétricas si están configuradas
                modified_gate_types = self.config.gate_types_A.copy()
            else:
                modified_gate_types = self.config.gate_types_A
                
            estado = self._create_enhanced_circuit_layer(
                estado, self.params.rotation_angles_A[layer], 
                modified_gate_types, 0
            )
            estado = self._apply_enhanced_entanglement(
                estado, range(self.num_qubits_A)
            )
        
        # Estado intermedio
        intermediate_asymmetry = self.asymmetry_analyzer.calculate_state_asymmetry(estado)
        
        # Circuito B
        for layer in range(self.config.circuit_B_layers):
            if 'RY_ASYM' in self.config.gate_types_B:
                modified_gate_types = self.config.gate_types_B.copy()
            else:
                modified_gate_types = self.config.gate_types_B
                
            estado = self._create_enhanced_circuit_layer(
                estado, self.params.rotation_angles_B[layer], 
                modified_gate_types, self.num_qubits_A
            )
            estado = self._apply_enhanced_entanglement(
                estado, range(self.num_qubits_A, self.total_qubits)
            )
        
        # Entrelazamiento inter-circuitos con acoplamiento personalizado
        for _ in range(self.config.entanglement_layers):
            for i in range(self.num_qubits_A):
                estado = self._apply_custom_coupling(
                    estado, i, i + self.num_qubits_A, 
                    self.config.coupling_strength
                )
        
        # Estado final
        final_asymmetry = self.asymmetry_analyzer.calculate_state_asymmetry(estado)
        
        # Guardar análisis de asimetría
        asymmetry_data = {
            'initial': initial_asymmetry,
            'intermediate': intermediate_asymmetry,
            'final': final_asymmetry
        }
        self.asymmetry_analyzer.asymmetry_history.append(asymmetry_data)
        
        # Guardar estado para análisis posterior
        self.quantum_states_history.append(estado.copy())
        
        # Medición estándar
        measurements = np.array([self._measure_qubit(estado, i) for i in range(self.total_qubits)])
        
        # Combinación clásica
        output = np.tanh(np.dot(self.params.classical_weights, measurements))
        
        return output, estado
    
    def analyze_quantum_dynamics(self) -> Dict[str, Any]:
        """
        Analiza la dinámica cuántica durante el entrenamiento
        """
        if not self.asymmetry_analyzer.asymmetry_history:
            return {}
        
        asymmetries = self.asymmetry_analyzer.asymmetry_history
        
        # Evolución de asimetría
        initial_asym = [a['initial'] for a in asymmetries]
        final_asym = [a['final'] for a in asymmetries]
        
        # Estadísticas
        analysis = {
            'asymmetry_evolution': {
                'initial_mean': np.mean(initial_asym),
                'final_mean': np.mean(final_asym),
                'asymmetry_change': np.mean(final_asym) - np.mean(initial_asym),
                'stability': np.std(final_asym)
            },
            'quantum_coherence': {
                'coherence_preservation': len([a for a in final_asym if a > self.config.asymmetry_threshold]) / len(final_asym),
                'max_asymmetry': np.max(final_asym),
                'min_asymmetry': np.min(final_asym)
            }
        }
        
        # Análisis de patrones de acoplamiento si hay suficientes mediciones
        if len(self.quantum_states_history) > 1:
            coupling_analysis = self.asymmetry_analyzer.analyze_coupling_patterns(
                [self._extract_measurements(state) for state in self.quantum_states_history[-10:]]
            )
            analysis['coupling_patterns'] = coupling_analysis
        
        return analysis
    
    def _extract_measurements(self, estado: np.ndarray) -> np.ndarray:
        """Extrae mediciones de un estado cuántico"""
        return np.array([self._measure_qubit(estado, i) for i in range(self.total_qubits)])
    
    def enhanced_fit(self, X: np.ndarray, y: np.ndarray, experiment_name: str):
        """
        Entrenamiento mejorado con análisis cuántico
        """
        # Usar forward pass mejorado en lugar del estándar
        original_forward = self.forward_pass
        self.forward_pass = self.enhanced_forward_pass
        
        try:
            # Llamar al método de entrenamiento original
            super().fit(X, y, experiment_name)
            
            # Análisis post-entrenamiento
            quantum_analysis = self.analyze_quantum_dynamics()
            
            # Guardar análisis cuántico en la base de datos
            if self.experiment_id and quantum_analysis:
                self._save_quantum_analysis(quantum_analysis)
            
            print(f"\n--- Análisis Cuántico ---")
            if 'asymmetry_evolution' in quantum_analysis:
                asym_evo = quantum_analysis['asymmetry_evolution']
                print(f"Cambio de asimetría: {asym_evo['asymmetry_change']:.6f}")
                print(f"Estabilidad: {asym_evo['stability']:.6f}")
            
            if 'quantum_coherence' in quantum_analysis:
                coherence = quantum_analysis['quantum_coherence']
                print(f"Preservación de coherencia: {coherence['coherence_preservation']:.3f}")
            
        finally:
            # Restaurar forward pass original
            self.forward_pass = original_forward
    
    def _save_quantum_analysis(self, analysis: Dict[str, Any]):
        """
        Guarda análisis cuántico en la base de datos
        """
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Crear tabla si no existe
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS quantum_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        experiment_id INTEGER,
                        analysis_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                    )
                ''')
                
                # Insertar análisis
                cursor.execute(
                    'INSERT INTO quantum_analysis (experiment_id, analysis_data) VALUES (?, ?)',
                    (self.experiment_id, json.dumps(analysis, default=str))
                )
                conn.commit()
                
        except Exception as e:
            print(f"Error guardando análisis cuántico: {e}")
    
    def visualize_quantum_evolution(self):
        """
        Visualiza la evolución cuántica durante el entrenamiento
        """
        if not self.asymmetry_analyzer.asymmetry_history:
            print("No hay datos de asimetría para visualizar")
            return
        
        asymmetries = self.asymmetry_analyzer.asymmetry_history
        epochs = range(len(asymmetries))
        
        initial_asym = [a['initial'] for a in asymmetries]
        intermediate_asym = [a['intermediate'] for a in asymmetries]
        final_asym = [a['final'] for a in asymmetries]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Evolución de asimetría
        ax1.plot(epochs, initial_asym, 'b-', label='Inicial', marker='o')
        ax1.plot(epochs, intermediate_asym, 'g-', label='Intermedio', marker='s')
        ax1.plot(epochs, final_asym, 'r-', label='Final', marker='^')
        ax1.axhline(y=self.config.asymmetry_threshold, color='k', linestyle='--', alpha=0.5, label='Umbral')
        ax1.set_title('Evolución de Asimetría Cuántica')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Asimetría')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distribución de asimetría final
        ax2.hist(final_asym, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax2.axvline(x=np.mean(final_asym), color='blue', linestyle='-', linewidth=2, label=f'Media: {np.mean(final_asym):.4f}')
        ax2.axvline(x=self.config.asymmetry_threshold, color='black', linestyle='--', alpha=0.5, label='Umbral')
        ax2.set_title('Distribución Asimetría Final')
        ax2.set_xlabel('Asimetría')
        ax2.set_ylabel('Frecuencia')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_quantum_data(self, filename: str = "enhanced_qnn_data.npz"):
        """
        Exporta datos cuánticos para análisis externo
        """
        if not self.asymmetry_analyzer.asymmetry_history:
            print("No hay datos cuánticos para exportar")
            return
        
        try:
            # Preparar datos para exportación
            asymmetry_data = np.array([[a['initial'], a['intermediate'], a['final']] 
                                     for a in self.asymmetry_analyzer.asymmetry_history])
            
            quantum_states = np.array([np.abs(state)**2 for state in self.quantum_states_history]) if self.quantum_states_history else np.array([])
            
            # Análisis cuántico completo
            analysis = self.analyze_quantum_dynamics()
            
            np.savez(filename,
                    asymmetry_evolution=asymmetry_data,
                    quantum_states_probabilities=quantum_states,
                    analysis_data=analysis,
                    config_asymmetry_factor=self.config.asymmetry_factor,
                    config_coupling_strength=self.config.coupling_strength,
                    config_pattern=self.config.custom_rotation_pattern)
            
            print(f"Datos cuánticos exportados a {filename}")
            
        except Exception as e:
            print(f"Error exportando datos cuánticos: {e}")

# Función de demostración
def demo_enhanced_qnn():
    """
    Demostración de la QNN mejorada con componentes cuánticos personalizados
    """
    print("=== Demo: QNN con Componentes Cuánticos Personalizados ===\n")
    
    # Cargar datos
    data = load_breast_cancer()
    X, y = data.data[:, :10], data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Configuración con componentes personalizados
    enhanced_config = CustomQuantumConfig(
        circuit_A_layers=2,
        circuit_B_layers=2,
        gate_types_A=['RY_ASYM', 'RZ'],  # Incluir puertas asimétricas
        gate_types_B=['RY', 'RZ'],
        optimizer='adam',
        learning_rate=0.02,
        epochs=15,
        batch_size=16,
        asymmetry_factor=0.05,
        coupling_strength=0.1,
        custom_rotation_pattern='asymmetric',
        enable_asymmetry_analysis=True,
        asymmetry_threshold=0.1
    )
    
    # Crear y entrenar modelo mejorado
    print("Creando QNN mejorada...")
    enhanced_qnn = EnhancedQNN(config=enhanced_config, num_features=X_train.shape[1])
    
    print("Entrenando con análisis cuántico...")
    start_time = time.time()
    enhanced_qnn.enhanced_fit(X_train, y_train, experiment_name="Enhanced_QNN_Demo")
    training_time = time.time() - start_time
    
    # Evaluación
    test_metrics = enhanced_qnn.evaluate(X_test, y_test)
    print(f"\nResultados:")
    print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")
    print(f"Precisión en prueba: {test_metrics['accuracy']:.4f}")
    
    # Visualizaciones
    print("\nGenerando visualizaciones...")
    enhanced_qnn.visualize_training()
    enhanced_qnn.visualize_quantum_evolution()
    
    # Exportar datos
    enhanced_qnn.export_quantum_data("demo_enhanced_qnn_data.npz")
    
    return enhanced_qnn

if __name__ == "__main__":
    # Ejecutar demostración
    enhanced_model = demo_enhanced_qnn()
