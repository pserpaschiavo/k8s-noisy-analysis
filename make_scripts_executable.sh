#!/bin/bash
# Make the pipeline scripts executable

chmod +x run_pipeline.py
chmod +x run_pipeline_with_experiment.py
chmod +x run_pipeline_3_rounds.py
chmod +x src/run_unified_pipeline.py
chmod +x debug_experiment_folder.py
chmod +x test_experiment_folder.py

echo "Pipeline scripts are now executable."
echo "Usage examples:"
echo "  ./run_pipeline.py --config config/pipeline_config.yaml"
echo "  ./run_pipeline_with_experiment.py --config config/pipeline_config_3rounds.yaml"
echo "  ./run_pipeline_3_rounds.py"
