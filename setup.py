#!/usr/bin/env python3
"""
Setup script for QuoreMind Framework
====================================

Framework modular de alto nivel para Aprendizaje Automático Cuántico (QML)
con enfoque en resiliencia y auditabilidad física en entornos ruidosos.

Instalación:
    pip install -e .
    
O para desarrollo:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages
import os
import sys
from pathlib import Path

# Leer el archivo README
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')

# Información del proyecto
NAME = "quoremind"
VERSION = "1.0.0"
AUTHOR = "tlacaelel666"
EMAIL = "your.email@example.com"  # Actualizar con tu email
DESCRIPTION = "Framework modular para Quantum Machine Learning con QEC y lógica Bayesiana"
URL = "https://github.com/tlacaelel666/QNN-hybrid"

# Dependencias principales
INSTALL_REQUIRES = [
    # Computación científica básica
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "matplotlib>=3.4.0",
    
    # Machine Learning
    "scikit-learn>=1.0.0",
    "tensorflow>=2.8.0",
    "tensorflow-probability>=0.15.0",
    
    # Visualización interactiva
    "plotly>=5.0.0",
    "dash>=2.0.0",
    "kaleido>=0.2.1",  # Para exportar gráficos plotly
    
    # Base de datos y persistencia
    "sqlalchemy>=1.4.0",
    "pandas>=1.3.0",
    
    # Utilidades
    "tqdm>=4.60.0",
    "click>=8.0.0",
    "pyyaml>=5.4.0",
    "python-dateutil>=2.8.0",
    
    # Computación cuántica (simuladores)
    "qiskit>=0.39.0",
    "cirq>=0.14.0",
    
    # Análisis estadístico avanzado
    "statsmodels>=0.13.0",
    "seaborn>=0.11.0",
    
    # Optimización
    "optuna>=3.0.0",
    "hyperopt>=0.2.7",
]

# Dependencias para desarrollo
DEV_REQUIRES = [
    "pytest>=6.2.0",
    "pytest-cov>=2.12.0",
    "pytest-xdist>=2.3.0",
    "black>=21.0.0",
    "flake8>=3.9.0",
    "mypy>=0.910",
    "isort>=5.9.0",
    "pre-commit>=2.15.0",
    "sphinx>=4.1.0",
    "sphinx-rtd-theme>=0.5.2",
    "notebook>=6.4.0",
    "jupyterlab>=3.1.0",
    "ipywidgets>=7.6.0",
]

# Dependencias para documentación
DOCS_REQUIRES = [
    "sphinx>=4.1.0",
    "sphinx-rtd-theme>=0.5.2",
    "sphinxcontrib-napoleon>=0.7",
    "myst-parser>=0.15.0",
    "nbsphinx>=0.8.0",
]

# Dependencias completas (todo incluido)
ALL_REQUIRES = INSTALL_REQUIRES + DEV_REQUIRES + DOCS_REQUIRES

# Clasificadores del proyecto
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

# Keywords para búsqueda
KEYWORDS = [
    "quantum computing",
    "machine learning", 
    "quantum neural networks",
    "error correction",
    "bayesian inference",
    "qml",
    "quantum algorithms",
    "parameter shift rule",
    "mahalanobis distance",
    "von neumann entropy"
]

def check_python_version():
    """Verifica que la versión de Python sea compatible."""
    if sys.version_info < (3, 8):
        raise RuntimeError(
            "QuoreMind requiere Python 3.8 o superior. "
            f"Versión actual: {sys.version_info.major}.{sys.version_info.minor}"
        )

def get_scripts():
    """Define los scripts de línea de comandos."""
    return [
        'bin/quoremind-run=main:main',
        'bin/quoremind-analyze=analysis:main',
        'bin/quoremind-benchmark=benchmark:main',
    ]

def get_entry_points():
    """Define los puntos de entrada del paquete."""
    return {
        'console_scripts': [
            'quoremind=main:main',
            'qm-analyze=scripts.analyze:main',
            'qm-benchmark=scripts.benchmark:main',
            'qm-visualize=scripts.visualize:main',
        ],
        'quoremind.optimizers': [
            'adam=quantum_nn:AdamOptimizer',
            'sgd=quantum_nn:SGDOptimizer',
            'rmsprop=quantum_nn:RMSpropOptimizer',
        ],
        'quoremind.error_models': [
            'depolarizing=quantum_error_correction_fixed:DepolarizingError',
            'bit_flip=quantum_error_correction_fixed:BitFlipError',
            'phase_flip=quantum_error_correction_fixed:PhaseFlipError',
        ]
    }

def create_manifest_in():
    """Crea el archivo MANIFEST.in si no existe."""
    manifest_content = """
include README.md
include LICENSE
include requirements.txt
include requirements-dev.txt
recursive-include quoremind *.py
recursive-include examples *.py *.ipynb
recursive-include docs *.rst *.md
recursive-include tests *.py
global-exclude *.pyc
global-exclude __pycache__
global-exclude .git*
global-exclude .DS_Store
"""
    
    manifest_path = Path("MANIFEST.in")
    if not manifest_path.exists():
        manifest_path.write_text(manifest_content.strip())
        print("✅ Archivo MANIFEST.in creado")

def create_requirements_files():
    """Crea archivos de requirements si no existen."""
    
    # requirements.txt (producción)
    req_path = Path("requirements.txt")
    if not req_path.exists():
        req_path.write_text('\n'.join(INSTALL_REQUIRES))
        print("✅ Archivo requirements.txt creado")
    
    # requirements-dev.txt (desarrollo)
    dev_req_path = Path("requirements-dev.txt")
    if not dev_req_path.exists():
        dev_content = '\n'.join(["-r requirements.txt"] + DEV_REQUIRES)
        dev_req_path.write_text(dev_content)
        print("✅ Archivo requirements-dev.txt creado")

def create_project_structure():
    """Crea la estructura básica del proyecto si no existe."""
    
    directories = [
        "quoremind",
        "tests", 
        "examples",
        "docs",
        "scripts",
        "data",
        "notebooks",
        "bin"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py en directorios de Python
            if directory in ["quoremind", "tests", "scripts"]:
                init_file = dir_path / "__init__.py"
                if not init_file.exists():
                    init_content = f'"""QuoreMind {directory.capitalize()} module."""\n'
                    init_file.write_text(init_content)
    
    print("✅ Estructura de directorios verificada/creada")

def setup_configuration():
    """Ejecuta configuraciones adicionales."""
    print("🔧 Configurando proyecto QuoreMind...")
    
    check_python_version()
    create_project_structure()
    create_requirements_files()
    create_manifest_in()
    
    print("✅ Configuración completa")

if __name__ == "__main__":
    setup_configuration()

# Configuración principal de setuptools
setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    project_urls={
        "Documentation": f"{URL}/wiki",
        "Source Code": URL,
        "Bug Tracker": f"{URL}/issues",
        "Changelog": f"{URL}/releases",
        "Discussion": f"{URL}/discussions",
    },
    
    # Configuración de paquetes
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    package_dir={"": "."},
    
    # Incluir archivos de datos
    package_data={
        "quoremind": [
            "data/*.json",
            "data/*.yaml", 
            "configs/*.yaml",
            "templates/*.html",
        ],
    },
    include_package_data=True,
    
    # Dependencias
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "docs": DOCS_REQUIRES,
        "all": ALL_REQUIRES,
        "quantum": [
            "qiskit>=0.39.0",
            "cirq>=0.14.0",
            "pennylane>=0.25.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.8.0",
            "cupy>=10.0.0",
        ],
        "visualization": [
            "plotly>=5.0.0",
            "dash>=2.0.0",
            "bokeh>=2.4.0",
        ]
    },
    
    # Metadatos del proyecto
    classifiers=CLASSIFIERS,
    keywords=" ".join(KEYWORDS),
    license="Apache2.0",
    
    # Puntos de entrada
    entry_points=get_entry_points(),
    
    # Configuración adicional
    zip_safe=False,
    platforms=["any"],
    
    # Configuración de tests
    test_suite="tests",
    tests_require=DEV_REQUIRES,
    
    # Configuración de comandos personalizados
    cmdclass={},
)
