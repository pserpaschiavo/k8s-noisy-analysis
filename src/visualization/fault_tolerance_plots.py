"""
Module: visualization/fault_tolerance_plots.py
Description: Provides plotting functions for fault tolerance analysis.
"""
import os
import matplotlib.pyplot as plt
from typing import Dict, Any

def plot_fault_tolerance_analysis(results: Dict[str, Any], output_dir: str):
    """
    Generates and saves plots for fault tolerance analysis.
    """
    scenarios = list(results.keys())
    recovery_times = [res['recovery_time'] for res in results.values()]
    success_rates = [res['success_rate'] for res in results.values()]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot recovery times
    ax1.bar(scenarios, recovery_times, color='skyblue')
    ax1.set_ylabel('Recovery Time (s)')
    ax1.set_title('Recovery Time per Fault Scenario')

    # Plot success rates
    ax2.bar(scenarios, success_rates, color='lightgreen')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Success Rate per Fault Scenario')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "fault_tolerance_summary.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved fault tolerance plot to {plot_path}")
