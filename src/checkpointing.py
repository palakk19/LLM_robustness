import json
import os

from src.config import CHECKPOINT_FILE
from src.utils import debug_print

# ============================================================
# CHECKPOINTING
# ============================================================
def _fresh_checkpoint():
    return {
        "baseline_cache": {},
        "level_results": {
            "0": [],
            "1": [],
            "2": [],
            "3": [],
        }
    }

def load_checkpoint(path=CHECKPOINT_FILE):
    if not os.path.exists(path):
        debug_print("CKPT", f"No checkpoint found at {path}. Starting fresh.")
        return _fresh_checkpoint()

    try:
        with open(path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)

        if "baseline_cache" not in ckpt:
            ckpt["baseline_cache"] = {}
        if "level_results" not in ckpt:
            ckpt["level_results"] = {"0": [], "1": [], "2": [], "3": []}

        for k in ["0", "1", "2", "3"]:
            ckpt["level_results"].setdefault(k, [])

        debug_print(
            "CKPT",
            f"Loaded checkpoint: "
            f"{len(ckpt['baseline_cache'])} baselines, "
            f"L0={len(ckpt['level_results']['0'])}, "
            f"L1={len(ckpt['level_results']['1'])}, "
            f"L2={len(ckpt['level_results']['2'])}, "
            f"L3={len(ckpt['level_results']['3'])}"
        )
        return ckpt

    except Exception as e:
        debug_print("CKPT", f"Failed to load checkpoint ({e}). Starting fresh.")
        return _fresh_checkpoint()

def save_checkpoint(checkpoint, path=CHECKPOINT_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)