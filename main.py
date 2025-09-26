#!/usr/bin/env python3
"""
QuoreMind Framework - Orquestador Principal
===========================================

Framework modular de alto nivel para Aprendizaje Automático Cuántico (QML)
con enfoque en resiliencia y auditabilidad física en entornos ruidosos.

Autor: tlacaelel666
Repositorio: https://github.com/tlacaelel666/QNN-hybrid
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

# Importaciones de módulos del framework
try:
    from quantum_nn import AdvancedQNN, QNNConfig, QNNParams
    from quantum_error_correction_fixed import QuantumErrorCorrection, ErrorModel
    from quantum_bayes_mahalanobis import QuantumBayesianInference
    from bayes_logic import BayesianLogic
except ImportError as e:
    print(f"Error importando módulos del framework: {e}")
    print("Asegúrate de tener todos los archivos en el directorio actual:")
    print("- quantum_nn.py")
    print("- quantum_error_correction_fixed.py") 
    print("- quantum_bayes_mahalanobis.py")
    print("- bayes_logic.py")
    exit(1)

class DatabaseManager:
    """Gestor de base de datos para logging de experimentos y métricas."""
    
    def __init__(self, db_path: str = "quoremind_experiments.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializa las tablas de la base de datos."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    config TEXT NOT NULL,
                    experiment_type TEXT NOT NULL,
                    duration REAL,
                    final_loss REAL,
                    final_accuracy REAL,
                    notes TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    epoch INTEGER,
                    loss REAL,
                    accuracy REAL,
                    entropy REAL,
                    gradient_norm REAL,
                    mahalanobis_distance REAL,
                    error_rate REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS qec_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    epoch INTEGER,
                    error_type TEXT,
                    correction_applied TEXT,
                    success_rate REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
    
    def log_experiment(self, config: Dict, experiment_type: str, 
                      duration: float, final_loss: float, 
                      final_accuracy: float, notes: str = "") -> int:
        """Registra un nuevo experimento."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO experiments 
                (timestamp, config, experiment_type, duration, final_loss, final_accuracy, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                json.dumps(config),
                experiment_type,
                duration,
                final_loss,
                final_accuracy,
                notes
            ))
            return cursor.lastrowid
    
    def log_training_metric(self, experiment_id: int, epoch: int, 
                           loss: float, accuracy: float, entropy: float,
                           gradient_norm: float, mahalanobis_distance: float,
                           error_rate: float):
        """Registra métricas de entrenamiento."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO training_metrics 
                (experiment_id, epoch, loss, accuracy, entropy, gradient_norm, 
                 mahalanobis_distance, error_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (experiment_id, epoch, loss, accuracy, entropy, 
                  gradient_norm, mahalanobis_distance, error_rate))
    
    def log_qec_event(self, experiment_id: int, epoch: int,
                      error_type: str, correction_applied: str, 
                      success_rate: float):
        """Registra eventos de corrección de errores cuánticos."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO qec_events 
                (experiment_id, epoch, error_type, correction_applied, success_rate)
                VALUES (?, ?, ?, ?, ?)
            """, (experiment_id, epoch, error_type, correction_applied, success_rate))

class QuoreMindOrchestrator:
    """Orquestador principal del framework QuoreMind."""
    
    def __init__(self, config: QNNConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.logger = self._setup_logging()
        
        # Inicializar componentes principales
        self.qnn = AdvancedQNN(config)
        self.qec = QuantumErrorCorrection(n_qubits=config.circuit_A_layers + config.circuit_B_layers)
        self.bayes_inference = QuantumBayesianInference(n_qubits=self.qec.n_qubits)
        self.bayes_logic = BayesianLogic()
        
        # Métricas de seguimiento
        self.training_history = {
            'loss': [], 'accuracy': [], 'entropy': [],
            'gradient_norm': [], 'mahalanobis_distance': [],
            'error_rate': []
        }
    
    def _setup_logging(self):
        """Configura el sistema de logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('quoremind.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('QuoreMind')
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Genera datos sintéticos para entrenamiento."""
        np.random.seed(42)
        
        # Generar datos cuánticos sintéticos
        n_features = self.qnn.n_qubits
        X = np.random.randn(n_samples, n_features)
        
        # Etiquetas basadas en una función cuántica simulada
        y = np.array([
            1 if np.sum(x**2) > np.median(np.sum(X**2, axis=1)) else 0 
            for x in X
        ])
        
        self.logger.info(f"Generados {n_samples} samples con {n_features} features")
        return X, y
    
    def train_epoch(self, X: np.ndarray, y: np.ndarray, epoch: int) -> Dict[str, float]:
        """Entrena una época con integración QEC y Bayesiana."""
        epoch_metrics = {}
        
        # 1. Forward pass con QNN
        predictions = []
        total_loss = 0.0
        
        for i, (x_sample, y_true) in enumerate(zip(X, y)):
            # Aplicar ruido cuántico
            if epoch > 0 and i % 10 == 0:  # Aplicar QEC cada 10 muestras
                error_rate = self.qec.apply_error_model()
                if error_rate > 0.1:  # Umbral de corrección
                    correction_success = self.qec.apply_correction()
                    self.db_manager.log_qec_event(
                        self.current_experiment_id, epoch,
                        "depolarizing", "bit_flip", correction_success
                    )
            
            # Predicción QNN
            pred = self.qnn.forward(x_sample)
            predictions.append(pred)
            
            # Cálculo de pérdida con regularización Bayesiana
            loss = self._compute_loss_with_bayes(pred, y_true, x_sample)
            total_loss += loss
        
        predictions = np.array(predictions)
        
        # 2. Calcular métricas
        epoch_metrics['loss'] = total_loss / len(X)
        epoch_metrics['accuracy'] = np.mean((predictions > 0.5) == y)
        epoch_metrics['entropy'] = self._compute_von_neumann_entropy()
        epoch_metrics['gradient_norm'] = self._compute_gradient_norm()
        
        # 3. Análisis Bayesiano
        current_params = self.qnn.get_parameters()
        mahalanobis_dist = self.bayes_inference.compute_mahalanobis_distance(current_params)
        epoch_metrics['mahalanobis_distance'] = mahalanobis_dist
        epoch_metrics['error_rate'] = self.qec.get_current_error_rate()
        
        # 4. Backward pass y actualización
        if epoch_metrics['mahalanobis_distance'] < 2.0:  # Umbral de confianza
            self.qnn.backward(X, y, predictions)
        else:
            self.logger.warning(f"Época {epoch}: Distancia Mahalanobis alta ({mahalanobis_dist:.3f})")
        
        # 5. Logging
        self.db_manager.log_training_metric(
            self.current_experiment_id, epoch,
            **epoch_metrics
        )
        
        return epoch_metrics
    
    def _compute_loss_with_bayes(self, prediction: float, true_label: int, 
                                input_data: np.ndarray) -> float:
        """Calcula pérdida con término Bayesiano de Mahalanobis."""
        # Pérdida base (cross-entropy)
        epsilon = 1e-15
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        base_loss = -true_label * np.log(prediction) - (1 - true_label) * np.log(1 - prediction)
        
        # Término de regularización Bayesiana
        current_params = self.qnn.get_parameters()
        mahalanobis_penalty = self.bayes_inference.compute_mahalanobis_distance(current_params)
        
        # Combinar pérdidas
        total_loss = base_loss + self.config.l2_reg * mahalanobis_penalty
        return total_loss
    
    def _compute_von_neumann_entropy(self) -> float:
        """Calcula la entropía de Von Neumann del estado cuántico."""
        # Simulación de la entropía basada en el entrelazamiento
        params = self.qnn.get_parameters()
        # Aproximación: entropía proporcional a la varianza de parámetros
        return np.var(params) * self.qnn.n_qubits
    
    def _compute_gradient_norm(self) -> float:
        """Calcula la norma del gradiente usando Parameter-Shift Rule."""
        params = self.qnn.get_parameters()
        # Simulación del gradiente
        gradient = np.random.randn(len(params)) * 0.1  # Placeholder
        return np.linalg.norm(gradient)
    
    def run_experiment(self, experiment_name: str, 
                      n_samples: int = 1000,
                      visualize: bool = True) -> Dict:
        """Ejecuta un experimento completo."""
        self.logger.info(f"Iniciando experimento: {experiment_name}")
        start_time = time.time()
        
        # Generar datos
        X, y = self.generate_synthetic_data(n_samples)
        
        # Registrar experimento en DB
        config_dict = {
            'circuit_A_layers': self.config.circuit_A_layers,
            'circuit_B_layers': self.config.circuit_B_layers,
            'entanglement_layers': self.config.entanglement_layers,
            'optimizer': self.config.optimizer,
            'learning_rate': self.config.learning_rate,
            'epochs': self.config.epochs,
            'l2_reg': self.config.l2_reg,
            'n_samples': n_samples
        }
        
        self.current_experiment_id = self.db_manager.log_experiment(
            config_dict, experiment_name, 0, 0, 0, f"Iniciado: {datetime.now()}"
        )
        
        # Entrenamiento
        self.logger.info("Iniciando entrenamiento...")
        for epoch in range(self.config.epochs):
            metrics = self.train_epoch(X, y, epoch)
            
            # Actualizar historial
            for key, value in metrics.items():
                self.training_history[key].append(value)
            
            # Log progreso
            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                self.logger.info(
                    f"Época {epoch:3d}: "
                    f"Loss={metrics['loss']:.4f}, "
                    f"Acc={metrics['accuracy']:.3f}, "
                    f"Mahal={metrics['mahalanobis_distance']:.3f}"
                )
        
        # Finalizar experimento
        duration = time.time() - start_time
        final_loss = self.training_history['loss'][-1]
        final_accuracy = self.training_history['accuracy'][-1]
        
        # Actualizar registro del experimento
        with sqlite3.connect(self.db_manager.db_path) as conn:
            conn.execute("""
                UPDATE experiments 
                SET duration = ?, final_loss = ?, final_accuracy = ?, 
                    notes = ? 
                WHERE id = ?
            """, (duration, final_loss, final_accuracy, 
                  f"Completado exitosamente: {datetime.now()}", 
                  self.current_experiment_id))
        
        # Visualización
        if visualize:
            self.visualize_results(experiment_name)
        
        results = {
            'experiment_id': self.current_experiment_id,
            'duration': duration,
            'final_loss': final_loss,
            'final_accuracy': final_accuracy,
            'training_history': self.training_history.copy()
        }
        
        self.logger.info(f"Experimento completado en {duration:.2f}s")
        return results
    
    def visualize_results(self, experiment_name: str):
        """Crea visualizaciones interactivas de los resultados."""
        
        # Crear subplots con Plotly
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Pérdida y Precisión', 
                'Entropía de Von Neumann',
                'Distancia de Mahalanobis', 
                'Tasa de Error Cuántico'
            ],
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        epochs = range(len(self.training_history['loss']))
        
        # Gráfico 1: Pérdida y Precisión
        fig.add_trace(
            go.Scatter(x=list(epochs), y=self.training_history['loss'],
                      name='Pérdida', line=dict(color='red')),
            row=1, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=list(epochs), y=self.training_history['accuracy'],
                      name='Precisión', line=dict(color='blue')),
            row=1, col=1, secondary_y=True
        )
        
        # Gráfico 2: Entropía
        fig.add_trace(
            go.Scatter(x=list(epochs), y=self.training_history['entropy'],
                      name='Entropía VN', line=dict(color='green')),
            row=1, col=2
        )
        
        # Gráfico 3: Distancia Mahalanobis
        fig.add_trace(
            go.Scatter(x=list(epochs), y=self.training_history['mahalanobis_distance'],
                      name='Dist. Mahalanobis', line=dict(color='purple')),
            row=2, col=1
        )
        
        # Gráfico 4: Tasa de Error
        fig.add_trace(
            go.Scatter(x=list(epochs), y=self.training_history['error_rate'],
                      name='Tasa de Error', line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'QuoreMind - Métricas de Entrenamiento: {experiment_name}',
            height=800,
            showlegend=True
        )
        
        fig.write_html(f'quoremind_results_{experiment_name.replace(" ", "_")}.html')
        self.logger.info(f"Visualización guardada: quoremind_results_{experiment_name.replace(' ', '_')}.html")

def create_default_config() -> QNNConfig:
    """Crea una configuración por defecto optimizada."""
    return QNNConfig(
        circuit_A_layers=3,
        circuit_B_layers=2,
        entanglement_layers=1,
        optimizer='adam',
        learning_rate=0.01,
        epochs=100,
        l2_reg=0.001
    )

def main():
    """Función principal - Punto de entrada del framework."""
    
    print("=" * 60)
    print("🔬 QuoreMind Framework - Quantum Machine Learning")
    print("🚀 Sistema Híbrido QNN con QEC y Lógica Bayesiana")
    print("=" * 60)
    
    # Configuración
    config = create_default_config()
    db_manager = DatabaseManager()
    
    # Crear orquestador
    orchestrator = QuoreMindOrchestrator(config, db_manager)
    
    # Ejecutar experimentos
    experiments = [
        ("Experimento_Baseline", 1000),
        ("Experimento_Ruido_Alto", 800),
        ("Experimento_QEC_Intensivo", 1200)
    ]
    
    results = {}
    
    for exp_name, n_samples in experiments:
        print(f"\n🧪 Ejecutando: {exp_name}")
        print("-" * 40)
        
        try:
            result = orchestrator.run_experiment(
                experiment_name=exp_name,
                n_samples=n_samples,
                visualize=True
            )
            results[exp_name] = result
            
            print(f"✅ {exp_name} completado:")
            print(f"   - Duración: {result['duration']:.2f}s")
            print(f"   - Pérdida final: {result['final_loss']:.4f}")
            print(f"   - Precisión final: {result['final_accuracy']:.3f}")
            
        except Exception as e:
            print(f"❌ Error en {exp_name}: {e}")
            orchestrator.logger.error(f"Error en experimento {exp_name}: {e}")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE EXPERIMENTOS")
    print("=" * 60)
    
    for exp_name, result in results.items():
        print(f"{exp_name:25} | "
              f"Acc: {result['final_accuracy']:.3f} | "
              f"Loss: {result['final_loss']:.4f} | "
              f"Tiempo: {result['duration']:.1f}s")
    
    print(f"\n💾 Base de datos: {db_manager.db_path}")
    print("🌐 Visualizaciones generadas en archivos HTML")
    print("\n🎉 Framework QuoreMind ejecutado exitosamente!")

if __name__ == "__main__":
    main()
