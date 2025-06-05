#!/bin/bash
# Simple script to check the config file

CONFIG_PATH="config/pipeline_config.yaml"
echo "Checking file at: $CONFIG_PATH"

if [ -f "$CONFIG_PATH" ]; then
    echo "File exists. Contents:"
    cat "$CONFIG_PATH"
else
    echo "ERROR: File does not exist!"
fi
