#!/usr/bin/env python3
"""
Script: verify_presentation_materials.py
Description: Verify that all required materials for the presentation are available and ready.

This script checks that all necessary files exist and reports any issues or missing components.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def verify_presentation_materials():
    """Verify that all required presentation materials exist."""
    base_dir = Path("/home/phil/Projects/k8s-noisy-analysis")
    presentation_dir = base_dir / "outputs" / "presentation"
    
    # Track issues
    issues = []
    
    # Check if presentation directory exists
    if not presentation_dir.exists():
        issues.append("Main presentation directory not found")
        print("❌ CRITICAL: Presentation directory not found! You need to run 'python organize_presentation_visualizations.py' first.")
        return issues
    
    # Check subdirectories
    expected_dirs = [
        "01_phase_comparisons",
        "02_anomaly_detection",
        "03_correlation_analysis"
    ]
    
    for dir_name in expected_dirs:
        dir_path = presentation_dir / dir_name
        if not dir_path.exists():
            issues.append(f"Missing directory: {dir_name}")
            continue
        
        # Check if directory has content
        if not any(dir_path.iterdir()):
            issues.append(f"Directory {dir_name} is empty")
    
    # Check cross-correlation subdirectory
    ccf_dir = presentation_dir / "03_correlation_analysis" / "cross_correlation"
    if not ccf_dir.exists():
        issues.append("Missing cross-correlation directory")
    elif not any(ccf_dir.iterdir()):
        issues.append("Cross-correlation directory is empty")
    else:
        ccf_files = list(ccf_dir.glob("ccf_*.png"))
        if len(ccf_files) == 0:
            issues.append("No cross-correlation plots found")
        elif len(ccf_files) < 2:
            issues.append(f"Only {len(ccf_files)} cross-correlation plot found, expected at least 2")
    
    # Check key files
    expected_files = [
        "00_presentation_summary.md",
        "presentation_guide.md"
    ]
    
    for file_name in expected_files:
        file_path = presentation_dir / file_name
        if not file_path.exists():
            issues.append(f"Missing file: {file_name}")
        elif file_path.stat().st_size == 0:
            issues.append(f"File is empty: {file_name}")
    
    # Check correlation heatmaps
    expected_heatmaps = [
        "correlation_baseline.png",
        "correlation_attack.png",
        "correlation_recovery.png"
    ]
    
    for heatmap in expected_heatmaps:
        heatmap_path = presentation_dir / "03_correlation_analysis" / heatmap
        if not heatmap_path.exists():
            issues.append(f"Missing correlation heatmap: {heatmap}")
    
    # Check supporting documentation
    doc_dir = base_dir / "docs"
    expected_docs = [
        "correlacao_cruzada.md",
        "melhorias_implementadas.md"
    ]
    
    for doc in expected_docs:
        doc_path = doc_dir / doc
        if not doc_path.exists():
            issues.append(f"Missing documentation: {doc}")
    
    # Check work plan is updated
    work_plan = base_dir / "analysis_work_plan.md"
    if not work_plan.exists():
        issues.append("Missing analysis work plan")
    else:
        # Check if work plan is recent (updated in June)
        with open(work_plan, 'r') as f:
            content = f.read()
            if "Junho/2025" not in content and "June/2025" not in content:
                issues.append("Work plan may be outdated (no June 2025 date found)")
    
    # Check demo guide
    demo_guide = base_dir / "presentation_demo_guide.md"
    if not demo_guide.exists():
        issues.append("Missing presentation demo guide")
    
    return issues

def main():
    """Main function to verify presentation materials."""
    print("Verifying presentation materials...")
    
    # Add debug statement to track execution
    print("Debug: Starting verification...")
    
    try:
        issues = verify_presentation_materials()
        
        if not issues:
            print("✅ All presentation materials verified successfully!")
            print("You're ready for the presentation on June 5, 2025.")
        else:
            print(f"❌ Found {len(issues)} issues with presentation materials:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            print("\nPlease address these issues before the presentation.")
    except Exception as e:
        print(f"Error during verification: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if any("run 'python organize_presentation_visualizations.py'" in issue for issue in issues):
            print("\nSuggested fix: Run the following commands:")
            print("  python -m src.run_unified_pipeline --config config/pipeline_config.yaml")
            print("  python generate_presentation_insights.py")
            print("  python organize_presentation_visualizations.py")

if __name__ == "__main__":
    main()
