#!/usr/bin/env python3
"""
Script: check_visualization_outputs.py
Description: Simple script to check that visualization outputs exist
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('check_viz')

def count_files_by_extension(directory, extension='.png'):
    """Count files with the given extension in directory and its subdirectories."""
    count = 0
    for root, dirs, files in os.walk(directory):
        count += sum(1 for f in files if f.endswith(extension))
    return count

def check_outputs(output_dir):
    """Check visualization outputs in the specified directory."""
    logger.info(f"Checking outputs in {output_dir}")
    
    # Check if base directory exists
    if not os.path.exists(output_dir):
        logger.error(f"Output directory {output_dir} does not exist!")
        return
    
    # Check plots directory
    plots_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(plots_dir):
        logger.error(f"Plots directory {plots_dir} does not exist!")
        return
    
    # Check each expected plot category directory
    categories = [
        "descriptive", 
        "correlation", 
        "causality", 
        "anomaly_detection", 
        "sliding_window", 
        "phase_comparison"
    ]
    
    total_plots = 0
    logger.info("\nPlot counts by category:")
    
    for category in categories:
        category_dir = os.path.join(plots_dir, category)
        if os.path.exists(category_dir):
            plot_count = count_files_by_extension(category_dir, '.png')
            total_plots += plot_count
            logger.info(f"  - {category}: {plot_count} plots")
        else:
            logger.warning(f"  - {category}: directory not found")
    
    logger.info(f"\nTotal visualization outputs: {total_plots} plots")
    
    # Check if we have a reasonable number of plots
    if total_plots > 0:
        logger.info("✅ Visualization outputs found! The pipeline appears to have run successfully.")
    else:
        logger.error("❌ No visualization outputs found! The pipeline may have failed.")

def main():
    parser = argparse.ArgumentParser(description="Check visualization outputs exist")
    parser.add_argument("--output-dir", 
                        default="/home/phil/Projects/k8s-noisy-analysis/outputs/demo-experiment-3-rounds",
                        help="Directory containing visualization outputs")
    
    args = parser.parse_args()
    check_outputs(args.output_dir)

if __name__ == "__main__":
    main()
