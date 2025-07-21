#!/usr/bin/env python3
"""
Master script to verify center point accuracy by running all verification tools in sequence.
This script orchestrates:
1. Running the center analyzer to find problematic matches
2. Creating visualizations of centers on original images
3. Creating visualizations of centers on cropped images

Usage:
  python verify_all_centers.py [--tolerance TOLERANCE]
"""

import os
import sys
import argparse
import subprocess
import time

# Set paths
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '../../..'))

def run_script(script_name, args=None):
    """Run a Python script with optional arguments and return exit code."""
    cmd = [sys.executable, os.path.join(script_dir, script_name)]
    if args:
        cmd.extend(args)
        
    print(f"\n\n{'='*80}")
    print(f"Running {script_name} with arguments: {args}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=script_dir)
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"Finished {script_name} with exit code {result.returncode} in {elapsed_time:.1f} seconds")
    print(f"{'='*80}\n")
    
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description='Verify center point accuracy using multiple tools')
    parser.add_argument('--tolerance', type=int, default=500, 
                        help='Tolerance in pixels for bbox matching (default: 500)')
    parser.add_argument('--skip-combine', action='store_true',
                        help='Skip recombining the data with the specified tolerance')
    parser.add_argument('--fast', action='store_true',
                        help='Run only essential analysis, skip visualizations')
    args = parser.parse_args()
    
    # Step 1: Regenerate the combined data with the specified tolerance
    if not args.skip_combine:
        print("Step 1: Regenerating combined data with specified tolerance...")
        exit_code = run_script('combine_lengths_improved.py', ['--tolerance', str(args.tolerance)])
        if exit_code != 0:
            print("Error: Failed to regenerate combined data")
            sys.exit(1)
    else:
        print("Skipping data recombination as requested.")
    
    # Step 2: Run center match analysis
    print("Step 2: Analyzing center point matches...")
    exit_code = run_script('analyze_center_matches.py')
    if exit_code != 0:
        print("Warning: Center match analysis completed with errors")
    
    if args.fast:
        print("Skipping visualizations due to --fast flag")
        return
    
    # Step 3: Create visualizations on original images
    print("Step 3: Creating visualizations on original images...")
    exit_code = run_script('visualize_centers.py')
    if exit_code != 0:
        print("Warning: Original image visualization completed with errors")
    
    # Step 4: Create visualizations on cropped images
    print("Step 4: Creating visualizations on cropped images...")
    exit_code = run_script('visualize_centers_cropped.py')
    if exit_code != 0:
        print("Warning: Cropped image visualization completed with errors")
    
    # Summary
    print("\n" + "="*80)
    print("Center point verification process complete!")
    print(f"Analysis and visualization results are in:")
    print(f"- {os.path.join(base_dir, 'fifty_one/processed_data/center_analysis')}")
    print(f"- {os.path.join(base_dir, 'fifty_one/processed_data/center_verification')}")
    print(f"- {os.path.join(base_dir, 'fifty_one/processed_data/center_verification_cropped')}")
    print("="*80)

if __name__ == "__main__":
    main() 