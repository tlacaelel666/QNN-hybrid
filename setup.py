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
EMAIL = "tlacaelel666@example.com"  # Actualizar con tu email real
DESCRIPTION = "Framework modular para Quantum Machine Learning con QEC y lógica Bayesiana"
URL = "https://github.com/tlacaelel666/QNN-hybrid"

# Dependencias principales
INSTALL_REQUIRES = [
    # Computación científica básica
    "numpy>=1.21.0,<2.0.0",
    "scipy>=1.7.0,<2.0.0",
    "matplotlib>=3.4.0,<4.0.0",
    
    # Machine Learning
    "scikit-learn>=1.0.0,<2.0.0",
    "tensorflow>=2.8.0,<3.0.0",
    "tensorflow-probability>=0.15.0,<1.0.0",
    
    # Visualización interactiva
    "plotly>=5.0.0,<6.0.0",
    "dash>=2.0.0,<3.0.0",
    "kaleido>=0.2.1",  # Para exportar gráficos plotly
    
    # Base de datos y persistencia
    "sqlalchemy>=1.4.0,<2.0.0",
    "pandas>=1.3.0,<3.0.0",
    
    # Utilidades
    "tqdm>=4.60.0",
    "click>=8.0.0",
    "pyyaml>=5.4.0",
    "python-dateutil>=2.8.0",
    
    # Computación cuántica (simuladores)
    "qiskit>=0.39.0,<2.0.0",
    "qiskit-aer>=0.11.0",
    
    # Análisis estadístico avanzado
    "statsmodels>=0.13.0,<1.0.0",
    "seaborn>=0.11.0,<1.0.0",
    
    # Optimización
    "optuna>=3.0.0,<4.0.0",
    "hyperopt>=0.2.7,<1.0.0",
]

# Dependencias para desarrollo
DEV_REQUIRES = [
    "pytest>=6.2.0",
    "pytest-cov>=2.12.0",
    "pytest-xdist>=2.3.0",
    "pytest-mock>=3.6.0",
    "black>=21.0.0",
    "flake8>=3.9.0",
    "mypy>=0.910",
    "isort>=5.9.0",
    "pre-commit>=2.15.0",
    "notebook>=6.4.0",
    "jupyterlab>=3.1.0",
    "ipywidgets>=7.6.0",
    "pylint>=2.10.0",
    "coverage>=5.5",
    "bandit>=1.7.0",  # Seguridad
    "safety>=1.10.0",  # Vulnerabilidades
]

# Dependencias para documentación
DOCS_REQUIRES = [
    "sphinx>=4.1.0",
    "sphinx-rtd-theme>=0.5.2",
    "sphinxcontrib-napoleon>=0.7",
    "myst-parser>=0.15.0",
    "nbsphinx>=0.8.0",
    "sphinx-autodoc-typehints>=1.12.0",
    "sphinx-gallery>=0.10.0",
]

# Dependencias completas (todo incluido)
ALL_REQUIRES = INSTALL_REQUIRES + DEV_REQUIRES + DOCS_REQUIRES

# Clasificadores del proyecto
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Framework :: Jupyter",
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
    "von neumann entropy",
    "quantum simulation",
    "hybrid classical-quantum",
    "variational quantum eigensolver",
    "qaoa"
]

def check_python_version():
    """Verifica que la versión de Python sea compatible."""
    if sys.version_info < (3, 8):
        raise RuntimeError(
            "QuoreMind requiere Python 3.8 o superior. "
            f"Versión actual: {sys.version_info.major}.{sys.version_info.minor}"
        )

def get_entry_points():
    """Define los puntos de entrada del paquete."""
    return {
        'console_scripts': [
            'quoremind=quoremind.cli.main:main',
            'qm-analyze=quoremind.cli.analyze:main',
            'qm-benchmark=quoremind.cli.benchmark:main',
            'qm-visualize=quoremind.cli.visualize:main',
            'qm-train=quoremind.cli.train:main',
            'qm-evaluate=quoremind.cli.evaluate:main',
        ],
        'quoremind.optimizers': [
            'adam=quoremind.optimization.optimizers:AdamOptimizer',
            'sgd=quoremind.optimization.optimizers:SGDOptimizer',
            'rmsprop=quoremind.optimization.optimizers:RMSpropOptimizer',
            'quantum_natural_gradient=quoremind.optimization.optimizers:QuantumNaturalGradientOptimizer',
        ],
        'quoremind.error_models': [
            'depolarizing=quoremind.error_correction.models:DepolarizingError',
            'bit_flip=quoremind.error_correction.models:BitFlipError',
            'phase_flip=quoremind.error_correction.models:PhaseFlipError',
            'amplitude_damping=quoremind.error_correction.models:AmplitudeDampingError',
        ],
        'quoremind.quantum_layers': [
            'variational=quoremind.quantum_nn.layers:VariationalLayer',
            'embedding=quoremind.quantum_nn.layers:EmbeddingLayer',
            'measurement=quoremind.quantum_nn.layers:MeasurementLayer',
        ]
    }

def create_manifest_in():
    """Crea el archivo MANIFEST.in si no existe."""
    manifest_content = """
include README.md
include LICENSE
include CHANGELOG.md
include requirements*.txt
include pyproject.toml
include tox.ini
recursive-include quoremind *.py
recursive-include quoremind *.yaml *.json *.html *.css *.js
recursive-include examples *.py *.ipynb *.md
recursive-include docs *.rst *.md *.py
recursive-include tests *.py *.yaml *.json
recursive-include data *.csv *.json *.h5 *.npz
prune quoremind/**/__pycache__
global-exclude *.pyc
global-exclude __pycache__
global-exclude .git*
global-exclude .DS_Store
global-exclude *.egg-info
global-exclude .pytest_cache
global-exclude .mypy_cache
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
        "quoremind/core",
        "quoremind/quantum_nn", 
        "quoremind/error_correction",
        "quoremind/optimization",
        "quoremind/bayesian",
        "quoremind/visualization",
        "quoremind/cli",
        "quoremind/utils",
        "quoremind/data",
        "quoremind/configs",
        "tests",
        "tests/unit", 
        "tests/integration",
        "examples",
        "examples/notebooks",
        "docs",
        "docs/source",
        "scripts",
        "data",
        "notebooks"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py en directorios de Python
            if directory.startswith(("quoremind", "tests")):
                init_file = dir_path / "__init__.py"
                if not init_file.exists():
                    if directory == "quoremind":
                        init_content = '''"""
QuoreMind: Framework modular para Quantum Machine Learning
=========================================================

Framework de alto nivel para desarrollo de algoritmos de aprendizaje automático
cuántico con enfoque en resiliencia y auditabilidad física.
"""

__version__ = "1.0.0"
__author__ = "tlacaelel666"
__license__ = "Apache 2.0"

from .core import *
from .quantum_nn import *
from .error_correction import *

__all__ = [
    "QuantumNeuralNetwork",
    "ErrorCorrectionModule", 
    "BayesianOptimizer",
    "QuantumCircuitLayer",
    "MahalanobisDistance",
    "VonNeumannEntropy"
]
'''
                    else:
                        module_name = directory.split('/')[-1].replace('_', ' ').title()
                        init_content = f'"""QuoreMind {module_name} module."""\n'
                    
                    init_file.write_text(init_content)
    
    print("✅ Estructura de directorios verificada/creada")

def create_config_files():
    """Crea archivos de configuración del proyecto."""
    
    # pyproject.toml para herramientas modernas
    pyproject_content = """[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\\.pyi?$'
extend-exclude = '''
/(
  # directories
  \\.eggs
  | \\.git
  | \\.hg
  | \\.mypy_cache
  | \\.tox
  | \\.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["quoremind"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short --strict-markers"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "quantum: marks tests that require quantum simulators"
]

[tool.coverage.run]
source = ["quoremind"]
omit = [
    "*/tests/*",
    "*/test_*",
    "setup.py",
    "*/conftest.py"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:"
]
"""
    
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        pyproject_path.write_text(pyproject_content)
        print("✅ Archivo pyproject.toml creado")
    
    # .gitignore
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# QuoreMind specific
experiments/
models/
checkpoints/
logs/
*.h5
*.pkl
*.joblib
"""
    
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        gitignore_path.write_text(gitignore_content)
        print("✅ Archivo .gitignore creado")

def setup_configuration():
    """Ejecuta configuraciones adicionales."""
    print("🔧 Configurando proyecto QuoreMind...")
    
    check_python_version()
    create_project_structure()
    create_requirements_files()
    create_manifest_in()
    create_config_files()
    
    print("✅ Configuración completa")
    print("\n📋 Próximos pasos recomendados:")
    print("   1. git init")
    print("   2. git add .")
    print("   3. git commit -m 'Initial commit'")
    print("   4. pip install -e '.[dev]'")
    print("   5. pre-commit install")

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
        "CI/CD": f"{URL}/actions",
    },
    
    # Configuración de paquetes
    packages=find_packages(exclude=["tests*", "docs*", "examples*", "scripts*"]),
    package_dir={"": "."},
    
    # Incluir archivos de datos
    package_data={
        "quoremind": [
            "data/*.json",
            "data/*.yaml", 
            "data/*.csv",
            "configs/*.yaml",
            "configs/*.json",
            "templates/*.html",
            "templates/*.css",
            "templates/*.js",
        ],
    },
    include_package_data=True,
    
    # Dependencias
    python_requires=">=3.8,<4.0",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "docs": DOCS_REQUIRES,
        "all": ALL_REQUIRES,
        "quantum": [
            "qiskit>=0.39.0,<2.0.0",
            "pennylane>=0.25.0,<1.0.0",
            "qutip>=4.6.0,<5.0.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.8.0,<3.0.0",
            "cupy>=10.0.0,<13.0.0",
            "jax[gpu]>=0.3.0,<1.0.0",
        ],
        "visualization": [
            "plotly>=5.0.0,<6.0.0",
            "dash>=2.0.0,<3.0.0",
            "bokeh>=2.4.0,<4.0.0",
            "streamlit>=1.10.0,<2.0.0",
        ],
        "cloud": [
            "boto3>=1.20.0,<2.0.0",
            "google-cloud-storage>=2.0.0,<3.0.0",
            "azure-storage-blob>=12.0.0,<13.0.0",
        ]
    },
    
    # Metadatos del proyecto
    classifiers=CLASSIFIERS,
    keywords=" ".join(KEYWORDS),
    license="Apache-2.0",
    
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
