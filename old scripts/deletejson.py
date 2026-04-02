import json
import shutil

CHECKPOINT_FILE = "palak_final2/qwen5/run_checkpoint.json"
BACKUP_FILE = "palak_final2/qwen5/run_checkpoint_backup_before_L2_trim.json"

# Backup first
shutil.copy(CHECKPOINT_FILE, BACKUP_FILE)

with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
    checkpoint = json.load(f)

# 🔥 Keep only first 30 samples in L2
checkpoint["level_results"]["2"] = checkpoint["level_results"].get("2", [])[:30]

with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
    json.dump(checkpoint, f, indent=2, ensure_ascii=False)

print("Backup created:", BACKUP_FILE)
print("Trimmed L2 to first 30 samples.")

# Sanity check
for lvl in ["0", "1", "2", "3"]:
    print(f"L{lvl} count:", len(checkpoint["level_results"].get(lvl, [])))