# ICD-10 Robustness under Structured Medical Perturbations

This project evaluates the robustness of LLM-based ICD-10 coding under controlled perturbations in clinical discharge summaries. The goal is to analyze how sensitive models are to variations in medical terminology and whether they maintain consistent diagnostic predictions under structured perturbations.

---

## 📊 Dataset

This project uses a **preprocessed and stratified dataset** derived from MIMIC-IV discharge summaries.

### 📁 Included in Repository
data/stratified_dataset.csv

- Contains clinically cleaned and structured discharge summaries  
- Balanced across ICD-10 chapters  
- Used consistently across all experiments  

---

## 📂 Project Structure

- `src/config.py` — project configuration  
- `src/llm_connector.py` — RWTH Ollama server communication  
- `src/data_loader.py` — stratified dataset loading and MIMIC fallback loader  
- `src/perturbation.py` — medical entity extraction and controlled perturbation pipeline  
- `src/evaluation.py` — semantic similarity and baseline correctness logic  
- `src/prediction.py` — ICD prediction and description helpers  
- `src/analysis.py` — drift and stability analysis  
- `src/plotting.py` — visualization and plot generation  
- `src/reporting.py` — HTML report generation  
- `run_experiment.py` — main experiment runner  

---

## ⚙️ Methodology Overview

The pipeline evaluates robustness using **structured perturbations applied to medical entities**:

- **L0 — Synonyms** (surface-level changes)  
- **L1 — Clinical Broadening** (more general terms)  
- **L2 — Differential Substitution** (similar diseases)  
- **L3 — Cross-chapter Adversarial** (different systems)  

Each perturbed note is passed through the model, and predictions are compared against the original to measure **drift and stability**.

---

## 📊 Results

All experimental outputs are organized under the `results/` directory, separated by model.

### 📁 Directory Structure

results/  
├── llama/  
│   ├── drift_destination_plots/  
│   ├── main_plots/  
│   └── robustness_report_llama.py.html  
│  
├── medgemma/  
│   ├── drift_destination_plots/  
│   ├── main_plots/  
│   └── robustness_report_medgemma.py.html  
│  
├── qwen/  
│   ├── drift_destination_plots/  
│   ├── main_plots/  
│   └── robustness_report_qwen.py.html  

---

### 🔍 Explanation

Each model directory contains:

- **main_plots/**  
  Core evaluation plots including:
  - level-wise drift (L0–L3)  
  - chapter-wise vulnerability heatmaps  

- **drift_destination_plots/**  
  Visualizations showing *where predictions drift*, used to analyze:
  - systematic bias (e.g., musculoskeletal → injury)  
  - direction of model errors across ICD chapters  

- **robustness_report_*.html**  
  Detailed per-patient analysis including:
  - applied perturbations  
  - predicted ICD codes  
  - drift type (stable / subcategory / categorical)  
  - semantic similarity scores  

---

## 📈 Key Findings

- Even simple synonym changes (L0) can alter ICD predictions, indicating strong sensitivity to wording  
- Drift increases with perturbation severity (L0 → L3) across all models  
- Musculoskeletal and neoplasm chapters are most vulnerable due to overlapping terminology and reliance on precise wording  
- Injury and circulatory systems are most robust due to structured clinical representation   
- MedGemma (27B) shows highest robustness (lowest drift)  
- Llama 3.1 (8B) shows moderate performance  
- Qwen 2.5 (7B) is most sensitive to perturbations  

---

## 🚀 Run

```bash
python run_experiment.py