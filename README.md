# ICD-10 Robustness under Structured Medical Perturbations

This project evaluates the robustness of LLM-based ICD-10 coding under controlled perturbations in discharge summaries.

## Structure

- `src/config.py` - project configuration
- `src/llm_connector.py` - RWTH Ollama server communication
- `src/data_loader.py` - stratified dataset loading and MIMIC fallback loader
- `src/perturbation.py` - medical entity extraction and perturbation pipeline
- `src/evaluation.py` - semantic similarity and baseline correctness logic
- `src/prediction.py` - ICD prediction and description helpers
- `src/analysis.py` - drift and stability analysis
- `src/plotting.py` - plot generation
- `src/reporting.py` - HTML report generation
- `run_experiment.py` - main experiment runner

## Run

```bash
python run_experiment.py