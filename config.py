# utils.py
import logging
import os
from datetime import datetime

RESULTS_DIR = "resultados_diabetes"

class ColoredFormatter(logging.Formatter):
    """
    Um formatador de log personalizado que adiciona cor à saída do terminal
    com base no nível do log (INFO, WARNING, ERROR, etc.).
    """

    # Definimos os códigos de cor ANSI
    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    # Definimos o formato do log
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Mapeamos cada nível de log a uma cor e formato
    FORMATS = {
        logging.DEBUG: GREY + log_format + RESET,
        logging.INFO: GREEN + log_format + RESET,
        logging.WARNING: YELLOW + log_format + RESET,
        logging.ERROR: RED + log_format + RESET,
        logging.CRITICAL: BOLD_RED + log_format + RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logging():
    """Configura o sistema de logging para o projeto."""
    log_file_path = os.path.join(RESULTS_DIR, "logs")
    os.makedirs(log_file_path, exist_ok=True)
    log_file = os.path.join(log_file_path, f"log_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")

    # Remove handlers existentes para evitar duplicação de logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configuração base
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 2. Criamos um handler para os ficheiros de log (sem cor)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    # 3. Criamos um handler para o terminal (com cor)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())  # Usamos o nosso formatador colorido

    # Adicionamos os handlers ao logger principal
    # (Removendo a configuração básica para evitar duplicação)
    logger.propagate = False  # Impede que os logs subam para o logger raiz
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def create_project_directories():
    """Cria os diretórios necessários para salvar os resultados do projeto."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "modelos"), exist_ok=True)
    # ... (restante da função permanece igual)
    os.makedirs(os.path.join(RESULTS_DIR, "graficos", "confusao"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "graficos", "roc"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "graficos", "importancia"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "graficos", "distribuicao"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "graficos", "shap"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "graficos", "lime"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "history"), exist_ok=True)