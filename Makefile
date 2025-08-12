# Makefile for Gyroscopic Stabilized Quantum Engine

.PHONY: help install test clean run-demo lint

help:	## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:	## Install the package and dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev:	## Install with development dependencies
	pip install -r requirements.txt
	pip install -e ".[dev,recording,optimization]"

test:	## Run basic smoke tests
	@echo "Testing core physics imports..."
	@python3 -c "from pkgs.core_physics import build_tesseract_lattice; print('✓ Core physics OK')"
	@echo "Testing engine runtime..."
	@python3 -c "from pkgs.engine_runtime import RecursiveConformalComputing; print('✓ Engine runtime OK')"
	@echo "Testing engine service..."
	@python3 -c "import sys; sys.path.append('./apps/engine'); from engine_service import EngineService; print('✓ Engine service OK')"
	@echo "✅ All tests passed!"

test-integration:	## Run integration test with short simulation
	cd apps/engine && python3 -c "import sys; sys.path.extend(['.', '../..']); \
	from engine_service import EngineService, InitRequest, StepRequest; \
	engine = EngineService({}); \
	result = engine.init(InitRequest()); \
	step_result = engine.step(StepRequest(dt=0.01, r=0.1)); \
	print(f'✅ Integration test passed: ANE={step_result.ane_smear:.2e}')"

run-demo:	## Run a short demonstration simulation
	@echo "Running demonstration simulation (5 steps)..."
	@cd apps/engine && python3 -c "import yaml; \
	config = {'engine': {'r0': 0.5, 'm0': 1.0, 'enable_recorder': True}, \
	'physics': {'d_H': 3.12, 'C_RT': 91.64, 'kappa': 0.015, 'sigma_u': 0.004}, \
	'simulation': {'dt': 0.05, 'r_start': 0.1, 'r_end': 0.5, 'r_step': 0.1, 'max_steps': 5, 'save_interval': 2}, \
	'output': {'recordings_path': './demo_recordings', 'snapshot_path': './demo_snapshots'}}; \
	with open('demo_config.yaml', 'w') as f: yaml.dump(config, f)"
	cd apps/engine && python3 main.py --config demo_config.yaml
	@echo "✅ Demo completed! Check apps/engine/demo_* for outputs"

clean:	## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name "*.log" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
	rm -rf apps/engine/demo_* apps/engine/test_* 2>/dev/null || true

lint:	## Run basic code quality checks
	@echo "Checking for basic Python syntax..."
	@python3 -m py_compile pkgs/core_physics/*.py
	@python3 -m py_compile pkgs/engine_runtime/*.py  
	@python3 -m py_compile apps/engine/*.py
	@echo "✅ All Python files compile successfully"

package:	## Create distribution package
	python3 setup.py sdist bdist_wheel

install-package:	## Install from package
	pip install dist/*.whl

# Development shortcuts
dev-setup: install-dev test	## Complete development setup

quick-test: clean test		## Quick clean and test

full-test: clean test test-integration	## Run all tests