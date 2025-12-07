import os
import tempfile

# Get temp directory from environment variable or use system default
TEMP_BASE = os.environ.get('PRIVSYN_TEMP_DIR', tempfile.gettempdir())
TEMP_PRIVSYN = os.path.join(TEMP_BASE, "privsyn")   

# path related constant
RAW_DATA_PATH = "data/"
PROCESSED_DATA_PATH = os.path.join(TEMP_PRIVSYN, "temp_data", "processed_data")
SYNTHESIZED_RECORDS_PATH = os.path.join(TEMP_PRIVSYN, "temp_data", "synthesized_records")
MARGINAL_PATH = os.path.join(TEMP_PRIVSYN, "temp_data", "marginal")
DEPENDENCY_PATH = os.path.join(TEMP_PRIVSYN, "temp_data", "dependency")

ALL_PATH = [RAW_DATA_PATH, PROCESSED_DATA_PATH, SYNTHESIZED_RECORDS_PATH, MARGINAL_PATH, DEPENDENCY_PATH]

# config file path
TYPE_CONIFG_PATH = os.path.join(TEMP_PRIVSYN, "temp_data", "fields")

def set_temp_dir(custom_path):
    global TEMP_PRIVSYN, PROCESSED_DATA_PATH, SYNTHESIZED_RECORDS_PATH, MARGINAL_PATH, DEPENDENCY_PATH, TYPE_CONIFG_PATH, ALL_PATH
    
    TEMP_PRIVSYN = os.path.join(custom_path, "privsyn")
    PROCESSED_DATA_PATH = os.path.join(TEMP_PRIVSYN, "temp_data", "processed_data")
    SYNTHESIZED_RECORDS_PATH = os.path.join(TEMP_PRIVSYN, "temp_data", "synthesized_records")
    MARGINAL_PATH = os.path.join(TEMP_PRIVSYN, "temp_data", "marginal")
    DEPENDENCY_PATH = os.path.join(TEMP_PRIVSYN, "temp_data", "dependency")
    TYPE_CONIFG_PATH = os.path.join(TEMP_PRIVSYN, "temp_data", "fields")
    ALL_PATH = [RAW_DATA_PATH, PROCESSED_DATA_PATH, SYNTHESIZED_RECORDS_PATH, MARGINAL_PATH, DEPENDENCY_PATH]


