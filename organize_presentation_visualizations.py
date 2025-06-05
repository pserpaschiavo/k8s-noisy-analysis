#!/usr/bin/env python3
"""
Script: organize_presentation_visualizations.py
Description: Organize and prepare visualizations for the presentation.

This script helps organize the most important visualizations into a presentation folder
for easy access during the presentation.
"""

import os
import sys
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("organize_visuals")

def organize_visualizations(output_dir: str, presentation_dir: str) -> None:
    """
    Organize key visualizations into a presentation directory.
    
    Args:
        output_dir: The base output directory of the pipeline run
        presentation_dir: The directory where to save presentation visualizations
    """
    # Create presentation directory if it doesn't exist
    os.makedirs(presentation_dir, exist_ok=True)
    
    # Load insights if available
    insights_file = Path(output_dir) / "insights" / "presentation_insights.json"
    insights = {}
    if insights_file.exists():
        with open(insights_file, "r") as f:
            insights = json.load(f)
    
    # Create subdirectories for different visualization types
    phase_dir = os.path.join(presentation_dir, "01_phase_comparisons")
    os.makedirs(phase_dir, exist_ok=True)
    
    anomaly_dir = os.path.join(presentation_dir, "02_anomaly_detection")
    os.makedirs(anomaly_dir, exist_ok=True)
    
    correlation_dir = os.path.join(presentation_dir, "03_correlation_analysis")
    os.makedirs(correlation_dir, exist_ok=True)
    
    # Copy phase comparison plots
    phase_comparison_src = Path(output_dir) / "plots" / "phase_comparison"
    if phase_comparison_src.exists():
        for source_file in phase_comparison_src.glob("*.png"):
            dest_file = os.path.join(phase_dir, source_file.name)
            shutil.copy2(source_file, dest_file)
            logger.info(f"Copied {source_file} to {dest_file}")
    
    # Copy anomaly detection plots, focusing on those mentioned in insights
    anomaly_src = Path(output_dir) / "plots" / "anomaly_detection"
    if anomaly_src.exists():
        # If we have insights, prioritize the ones mentioned there
        if "anomaly_detections" in insights:
            for metric, anomalies in insights["anomaly_detections"].items():
                for anomaly in anomalies:
                    if "file_name" in anomaly:
                        source_file = Path(anomaly["file_name"])
                        if source_file.exists():
                            dest_file = os.path.join(anomaly_dir, source_file.name)
                            shutil.copy2(source_file, dest_file)
                            logger.info(f"Copied {source_file} to {dest_file}")
        else:
            # If no insights, just copy some representative anomaly plots
            # Focus on attack phase for key tenants
            for source_file in anomaly_src.glob("*_2_-_Attack_*.png"):
                dest_file = os.path.join(anomaly_dir, source_file.name)
                shutil.copy2(source_file, dest_file)
                logger.info(f"Copied {source_file} to {dest_file}")
    
    # Copy correlation heatmaps, one per phase
    correlation_src = Path(output_dir) / "plots" / "correlation"
    if correlation_src.exists():
        # Select a representative correlation heatmap for each phase
        for phase in ["1 - Baseline", "2 - Attack", "3 - Recovery"]:
            # Find a CPU usage correlation heatmap for this phase
            for source_file in correlation_src.glob(f"correlation_heatmap_cpu_usage_{phase}*.png"):
                dest_file = os.path.join(correlation_dir, f"correlation_{phase.split(' - ')[1].lower()}.png")
                shutil.copy2(source_file, dest_file)
                logger.info(f"Copied {source_file} to {dest_file}")
                break
                
        # Create a subdirectory for cross-correlation plots
        ccf_dir = os.path.join(correlation_dir, "cross_correlation")
        os.makedirs(ccf_dir, exist_ok=True)
        
        # Copy selected cross-correlation plots from the attack phase
        cross_correlation_src = Path(output_dir) / "plots" / "correlation" / "cross_correlation"
        if cross_correlation_src.exists():
            # Find CPU usage variability cross-correlation plots for attack phase
            ccf_count = 0
            for source_file in cross_correlation_src.glob(f"ccf_*_cpu_usage_variability_2 - Attack_*.png"):
                # Limit to a maximum of 4 plots to avoid cluttering
                if ccf_count >= 4:
                    break
                dest_file = os.path.join(ccf_dir, source_file.name)
                shutil.copy2(source_file, dest_file)
                logger.info(f"Copied {source_file} to {dest_file}")
                ccf_count += 1
    
    # Copy the presentation summary
    summary_src = Path(output_dir) / "insights" / "presentation_summary.md"
    if summary_src.exists():
        summary_dest = os.path.join(presentation_dir, "00_presentation_summary.md")
        shutil.copy2(summary_src, summary_dest)
        logger.info(f"Copied {summary_src} to {summary_dest}")
    
    logger.info(f"All presentation visualizations organized in {presentation_dir}")

def main():
    """Main function to organize visualizations for presentation."""
    try:
        # Define base output directory
        output_dir = "/home/phil/Projects/k8s-noisy-analysis/outputs/demo-experiment-1-round"
        
        # Define presentation directory
        presentation_dir = "/home/phil/Projects/k8s-noisy-analysis/outputs/presentation"
        
        # Organize visualizations
        print(f"Organizing visualizations from {output_dir} into {presentation_dir}")
        
        # Create presentation directory directly
        os.makedirs(presentation_dir, exist_ok=True)
        print(f"Created presentation directory: {presentation_dir}")
        
        organize_visualizations(output_dir, presentation_dir)
        
        print(f"Presentation materials prepared in {presentation_dir}")
        print("You can now use these files for your presentation tomorrow.")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
