#!/bin/bash
set -e

for config in experiments/*.yaml; do
    output_dir=$(python3 -c "import yaml; print(yaml.safe_load(open('$config'))['output']['output_dir'])")
    if [ -d "$output_dir" ]; then
        echo "Skipping $config (results already exist at $output_dir)"
        continue
    fi
    echo "=========================================="
    echo "Running: $config"
    echo "=========================================="
    python model.py --config "$config"
done

echo "All experiments complete."
