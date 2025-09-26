# QuoreMind Framework - Makefile
# ===============================
# Automatización de tareas comunes de desarrollo

.PHONY: help install install-dev test lint format clean docs run benchmark

# Variables
PYTHON := python3
PIP := pip3
PACKAGE_NAME := quoremind
TEST_PATH := tests/

# Colores para output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Ayuda por defecto
help: ## Muestra esta ayuda
	@echo "$(BLUE)QuoreMind Framework - Comandos Disponibles$(NC)"
	@echo "==========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Instalación
install: ## Instala el paquete en modo producción
	@echo "$(YELLOW)🚀 Instalando QuoreMind...$(NC)"
	$(PIP) install -e .
	@echo "$(GREEN)✅ Instalación completada$(NC)"

install-dev: ## Instala el paquete en modo desarrollo con todas las dependencias
	@echo "$(YELLOW)🔧 Instalando QuoreMind en modo desarrollo...$(NC)"
	$(PIP) install -e ".[dev,docs,all]"
	pre-commit install
	@echo "$(GREEN)✅ Instalación de desarrollo completada$(NC)"

install-quantum: ## Instala dependencias adicionales para computación cuántica
	@echo "$(YELLOW)⚛️ Instalando dependencias cuánticas...$(NC)"
	$(PIP) install -e ".[quantum]"
	@echo "$(GREEN)✅ Dependencias cuánticas instaladas$(NC)"

# Testing
test: ## Ejecuta todos los tests
	@echo "$(YELLOW)🧪 Ejecutando tests...$(NC)"
	pytest $(TEST_PATH) -v --cov=$(PACKAGE_NAME) --cov-report=html --cov-report=term
	@echo "$(GREEN)✅ Tests completados$(NC)"

test-fast: ## Ejecuta tests rápidos (sin cobertura)
	@echo "$(YELLOW)⚡ Ejecutando tests rápidos...$(NC)"
	pytest $(TEST_PATH) -x -v
	@echo "$(GREEN)✅ Tests rápidos completados$(NC)"

test-quantum: ## Ejecuta tests específicos de funcionalidad cuántica
	@echo "$(YELLOW)⚛️ Ejecutando tests cuánticos...$(NC)"
	pytest $(TEST_PATH) -k "quantum" -v
	@echo "$(GREEN)✅ Tests cuánticos completados$(NC)"

# Calidad de código
lint: ## Ejecuta linting del código
	@echo "$(YELLOW)🔍 Ejecutando linting...$(NC)"
	flake8 $(PACKAGE_NAME)/ main.py
	mypy $(PACKAGE_NAME)/ --ignore-missing-imports
	@echo "$(GREEN)✅ Linting completado$(NC)"

format: ## Formatea el código con black e isort
	@echo "$(YELLOW)🎨 Formateando código...$(NC)"
	black $(PACKAGE_NAME)/ main.py tests/ --line-length 88
	isort $(PACKAGE_NAME)/ main.py tests/ --profile black
	@echo "$(GREEN)✅ Formateo completado$(NC)"

format-check: ## Verifica el formateo sin modificar archivos
	@echo "$(YELLOW)👀 Verificando formateo...$(NC)"
	black $(PACKAGE_NAME)/ main.py tests/ --check --line-length 88
	isort $(PACKAGE_NAME)/ main.py tests/ --check-only --profile black
	@echo "$(GREEN)✅ Verificación de formateo completada$(NC)"

# Ejecución
run: ## Ejecuta el framework principal
	@echo "$(YELLOW)🚀 Ejecutando QuoreMind...$(NC)"
	$(PYTHON) main.py
	@echo "$(GREEN)✅ Ejecución completada$(NC)"

run-experiment: ## Ejecuta un experimento específico
	@echo "$(YELLOW)🧪 Ejecutando experimento personalizado...$(NC)"
	$(PYTHON) main.py --experiment custom --samples 1500 --epochs 150
	@echo "$(GREEN)✅ Experimento completado$(NC)"

benchmark: ## Ejecuta benchmarks de rendimiento
	@echo "$(YELLOW)📊 Ejecutando benchmarks...$(NC)"
	$(PYTHON) -m scripts.benchmark --full
	@echo "$(GREEN)✅ Benchmarks completados$(NC)"

# Documentación
docs: ## Genera la documentación
	@echo "$(YELLOW)📚 Generando documentación...$(NC)"
	cd docs && make html
	@echo "$(GREEN)✅ Documentación generada en docs/_build/html/$(NC)"

docs-serve: ## Sirve la documentación localmente
	@echo "$(YELLOW)🌐 Sirviendo documentación en http://localhost:8000$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

docs-clean: ## Limpia los archivos de documentación generados
	@echo "$(YELLOW)🧹 Limpiando documentación...$(NC)"
	cd docs && make clean
	@echo "$(GREEN)✅ Documentación limpiada$(NC)"

# Análisis y profiling
profile: ## Ejecuta profiling del código
	@echo "$(YELLOW)⏱️ Ejecutando profiling...$(NC)"
	$(PYTHON) -m cProfile -o profile_results.prof main.py
	$(PYTHON) -c "import pstats; pstats.Stats('profile_results.prof').sort_stats('cumulative').print_stats(20)"
	@echo "$(GREEN)✅ Profiling completado$(NC)"

analyze: ## Ejecuta análisis de código estático
	@echo "$(YELLOW)🔬 Ejecutando análisis de código...$(NC)"
	$(PYTHON) -m scripts.analyze --full-report
	@echo "$(GREEN)✅ Análisis completado$(NC)"

# Limpieza
clean: ## Limpia archivos temporales y cache
	@echo "$(YELLOW)🧹 Limpiando archivos temporales...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.prof" -delete
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	rm -rf *.db *.html *.log
	@echo "$(GREEN)✅ Limpieza completada$(NC)"

clean-all: clean docs-clean ## Limpieza completa incluyendo documentación
	@echo "$(GREEN)✅ Limpieza completa terminada$(NC)"

# Docker (opcional)
docker-build: ## Construye la imagen Docker
	@echo "$(YELLOW)🐳 Construyendo imagen Docker...$(NC)"
	docker build -t quoremind:latest .
	@echo "$(GREEN)✅ Imagen Docker construida$(NC)"

docker-run: ## Ejecuta el contenedor Docker
	@echo "$(YELLOW)🐳 Ejecutando contenedor Docker...$(NC)"
	docker run --rm -v $(PWD)/data:/app/data quoremind:latest
	@echo "$(GREEN)✅ Contenedor Docker ejecutado$(NC)"

# Distribución
build: clean ## Construye el paquete para distribución
	@echo "$(YELLOW)📦 Construyendo paquete...$(NC)"
	$(PYTHON) setup.py sdist bdist_wheel
	@echo "$(GREEN)✅ Paquete construido en dist/$(NC)"

upload-test: build ## Sube el paquete a TestPyPI
	@echo "$(YELLOW)📤 Subiendo a TestPyPI...$(NC)"
	twine upload --repository testpypi dist/*
	@echo "$(GREEN)✅ Paquete subido a TestPyPI$(NC)"

upload: build ## Sube el paquete a PyPI (PRODUCCIÓN)
	@echo "$(RED)⚠️ Subiendo a PyPI PRODUCCIÓN...$(NC)"
	@read -p "¿Estás seguro? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		twine upload dist/*; \
		echo "$(GREEN)✅ Paquete subido a PyPI$(NC)"; \
	else \
		echo "$(YELLOW)❌ Cancelado$(NC)"; \
	fi

# Desarrollo
dev-setup: ## Configuración inicial para desarrollo
	@echo "$(YELLOW)🔧 Configurando entorno de desarrollo...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	make install-dev
	@echo "$(GREEN)✅ Entorno de desarrollo configurado$(NC)"

pre-commit: ## Ejecuta verificaciones pre-commit
	@echo "$(YELLOW)🔍 Ejecutando verificaciones pre-commit...$(NC)"
	pre-commit run --all-files
	@echo "$(GREEN)✅ Verificaciones pre-commit completadas$(NC)"

# Información del sistema
info: ## Muestra información del sistema y dependencias
	@echo "$(BLUE)Sistema y Dependencias$(NC)"
	@echo "======================"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@echo "Directorio actual: $$(pwd)"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'No git')"
	@echo "Git commit: $$(git rev-parse --short HEAD 2>/dev/null || echo 'No git')"
	@echo ""
	@echo "$(BLUE)Estructura del proyecto:$(NC)"
	@find . -maxdepth 2 -name "*.py" | head -10
	@echo ""

# Ejemplos y demos
demo: ## Ejecuta una demostración completa del framework
	@echo "$(YELLOW)🎬 Ejecutando demostración...$(NC)"
	$(PYTHON) examples/demo.py --interactive
	@echo "$(GREEN)✅ Demostración completada$(NC)"

examples: ## Ejecuta todos los ejemplos
	@echo "$(YELLOW)📝 Ejecutando ejemplos...$(NC)"
	for example in examples/*.py; do \
		echo "Ejecutando $$example..."; \
		$(PYTHON) "$$example"; \
	done
	@echo "$(GREEN)✅ Ejemplos completados$(NC)"

# Verificación completa
check-all: format-check lint test ## Ejecuta todas las verificaciones
	@echo "$(GREEN)✅ Todas las verificaciones pasaron$(NC)"

# Instalación desde cero
fresh-install: clean ## Instalación limpia desde cero
	@echo "$(YELLOW)🔄 Instalación limpia...$(NC)"
	$(PIP) uninstall -y $(PACKAGE_NAME)
	make install-dev
	@echo "$(GREEN)✅ Instalación limpia completada$(NC)"
