# =========================================
# CONFIG
# =========================================
REPO_URL = https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git
REPO_DIR = SPRSound

VENV_DIR = venv
PYTHON = python3

# Cambia esto si tu entrypoint es distinto
MAIN_SCRIPT = main.py

REQUIREMENTS = \
	librosa \
	numpy \
	pandas \
	soundfile \
	kagglehub

# =========================================
# PHONY TARGETS
# =========================================
.PHONY: all clone venv install clean reclone run check

all: clone venv install

# =========================================
# CLONE
# =========================================
clone:
	@if [ ! -d "$(REPO_DIR)" ]; then \
		echo "[INFO] Cloning SPRSound repository..."; \
		git clone $(REPO_URL); \
	else \
		echo "[INFO] Repository already exists: $(REPO_DIR)"; \
	fi

# =========================================
# VENV (ROBUSTO)
# =========================================
venv:
	@if [ ! -f "$(VENV_DIR)/bin/activate" ]; then \
		echo "[INFO] Creating virtual environment..."; \
		rm -rf $(VENV_DIR); \
		$(PYTHON) -m venv $(VENV_DIR) || ( \
			echo "[WARN] Installing python3-venv..."; \
			sudo apt update && sudo apt install -y python3-venv; \
			$(PYTHON) -m venv $(VENV_DIR); \
		); \
	else \
		echo "[INFO] Virtual environment OK"; \
	fi

# =========================================
# INSTALL (SIN activate)
# =========================================
install: venv
	@echo "[INFO] Installing Python dependencies..."
	@$(VENV_DIR)/bin/pip install --upgrade pip
	@$(VENV_DIR)/bin/pip install $(REQUIREMENTS)

# =========================================
# CHECK ENTORNO
# =========================================
check:
	@echo "[INFO] Checking environment..."
	@$(VENV_DIR)/bin/python -c "import librosa, numpy, pandas, soundfile, kagglehub; print('All imports OK')"

# =========================================
# RUN PIPELINE
# =========================================
run:
	@echo "[INFO] Running pipeline..."
	@$(VENV_DIR)/bin/python $(MAIN_SCRIPT)

# =========================================
# LIMPIEZA
# =========================================
clean:
	@echo "[INFO] Cleaning project..."
	rm -rf $(REPO_DIR)
	rm -rf $(VENV_DIR)

# =========================================
# RECLONE TOTAL
# =========================================
reclone: clean all