import warnings

# ============================================================
# CONFIGURATION
# ============================================================
ORACLE_MODEL   = "llama3.1:8b"             # The Judge/Attacker (Generates & Validates)
TARGET_MODEL   = "qwen2.5:7b"              # Goes through levels of attack and is evaluated

API_URL        = "http://ollama.warhol.informatik.rwth-aachen.de/api/generate"
NOTES_PATH     = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
DIAGNOSES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/diagnoses_icd.csv.gz"
DICT_PATH      = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/d_icd_diagnoses.csv.gz"

FINAL_RESULTS  = "palak_final2/qwen6/final_robustness.csv"
HTML_REPORT    = "palak_final2/qwen6/robustness_report.html"
PLOT_FILE      = "palak_final2/qwen6/robustness_plots.png"
DEVICE         = "cpu"

BASELINE_SIM_THRESHOLD = 0.88

RANDOM_SEED = 42
TOTAL_EXPERIMENT_SAMPLES = None

CHECKPOINT_FILE  = "palak_final2/qwen6/run_checkpoint.json"
CHECKPOINT_EVERY = 10

# FIX: all perturbation rates set to 0.30, cap at 8 changes max
# MAX_CHANGES = 8
ATTACK_PERTURBATION_RATES = {0: 0.30, 1: 0.30, 2: 0.30, 3: 0.30}

# Minimum n per chapter to include in chapter analysis
MIN_CHAPTER_N = 20
MIN_ICD_N = 5
MAX_ICD_PLOT_COUNT = 10

ENTITY_WEIGHTS = {
    "DISEASE": 1.0,
    "CHEMICAL": 0.6,
    "ANATOMY": 0.4,
    "PROCEDURE": 0.5,
}

warnings.filterwarnings("ignore")