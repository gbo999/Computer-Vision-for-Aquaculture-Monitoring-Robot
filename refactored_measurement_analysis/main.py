#!/usr/bin/env python3
"""
Main entry point for prawn measurement analysis.

This script provides a clean command-line interface for running measurement analysis
with different configurations, replacing the original measurements_analysis.py.

Key improvements:
- Professional argument parsing with comprehensive help
- Input validation and error handling  
- Support for different measurement types and weight configurations
- Configurable output directories and ports
- Graceful session management
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .measurement_analysis import MeasurementAnalyzer
from .config import Config


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments with comprehensive validation.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Prawn Measurement Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --type carapace --weights car
  %(prog)s --type body --weights all --port 5160 --verbose
  %(prog)s --type carapace --weights kalkar --output-dir ./results --no-fiftyone
  
Measurement Types:
  carapace    Measure carapace length (start-carapace to eyes keypoints)
  body        Measure full body length (tail to rostrum keypoints)
  
Weight Types:
  car         Use car pond training weights
  kalkar      Use kalkar pond training weights  
  all         Use combined training weights
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--type', '-t',
        choices=['carapace', 'body'],
        required=True,
        help='Type of measurement to perform'
    )
    
    parser.add_argument(
        '--weights', '-w',
        choices=['car', 'kalkar', 'all'],
        required=True,
        help='Type of training weights to use for predictions'
    )
    
    # Optional arguments
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=None,
        help='Port number for FiftyOne app (default: auto-detect)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=None,
        help='Output directory for results (default: current directory)'
    )
    
    parser.add_argument(
        '--no-fiftyone',
        action='store_true',
        help='Skip FiftyOne dataset creation'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate parsed arguments and check prerequisites.
    
    Args:
        args: Parsed arguments
        
    Returns:
        True if validation passes, False otherwise
    """
    # Validate output directory
    if args.output_dir:
        if not args.output_dir.exists():
            try:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                if args.verbose:
                    print(f"Created output directory: {args.output_dir}")
            except Exception as e:
                print(f"Error: Cannot create output directory {args.output_dir}: {e}")
                return False
        
        if not args.output_dir.is_dir():
            print(f"Error: Output path {args.output_dir} is not a directory")
            return False
    
    # Validate port range
    if args.port and not (1024 <= args.port <= 65535):
        print(f"Error: Port {args.port} is not in valid range (1024-65535)")
        return False
    
    return True


def print_analysis_summary(analyzer: MeasurementAnalyzer) -> None:
    """
    Print analysis summary statistics.
    
    Args:
        analyzer: MeasurementAnalyzer instance with completed analysis
    """
    summary = analyzer.get_analysis_summary()
    
    if not summary:
        print("No analysis summary available")
        return
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Measurement Type: {summary['measurement_type']}")
    print(f"Weights Type: {summary['weights_type']}")
    print(f"Total Measurements: {summary['total_measurements']}")
    
    if 'pond_type_distribution' in summary:
        print(f"\nPond Distribution:")
        for pond, count in summary['pond_type_distribution'].items():
            print(f"  {pond}: {count}")
    
    if 'mean_error_percentage' in summary:
        print(f"\nError Statistics:")
        print(f"  Mean Error: {summary['mean_error_percentage']:.2f}%")
        print(f"  Median Error: {summary['median_error_percentage']:.2f}%")
        print(f"  Std Dev: {summary['std_error_percentage']:.2f}%")
        print(f"  Min Error: {summary['min_error_percentage']:.2f}%")
        print(f"  Max Error: {summary['max_error_percentage']:.2f}%")
    
    print("="*60)


def main() -> int:
    """
    Main entry point for the measurement analysis tool.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Parse and validate arguments
        args = parse_arguments()
        
        if not validate_arguments(args):
            return 1
        
        if args.verbose:
            print("Starting prawn measurement analysis...")
            print(f"Configuration: {args.type} measurements with {args.weights} weights")
        
        # Initialize analyzer
        analyzer = MeasurementAnalyzer(
            measurement_type=args.type,
            weights_type=args.weights,
            port=args.port,
            verbose=args.verbose
        )
        
        # Run analysis
        results_df, dataset_name = analyzer.run_analysis(
            output_dir=args.output_dir,
            create_fiftyone=not args.no_fiftyone
        )
        
        if results_df.empty:
            print("Analysis completed but no results generated")
            return 1
        
        # Print summary
        print_analysis_summary(analyzer)
        
        # Launch FiftyOne if requested
        if not args.no_fiftyone and dataset_name:
            print(f"\nFiftyOne dataset created: {dataset_name}")
            
            # Ask user if they want to launch the app
            try:
                response = input("\nLaunch FiftyOne app? (y/N): ").strip().lower()
                if response in ['y', 'yes']:
                    print(f"Launching FiftyOne app on port {analyzer.dataset_manager.port}...")
                    app = analyzer.launch_fiftyone_app()
                    if app:
                        print("FiftyOne app launched successfully!")
                        print("Press Ctrl+C to exit...")
                        try:
                            # Keep the script running while FiftyOne is active
                            app.wait()
                        except KeyboardInterrupt:
                            print("\nShutting down FiftyOne app...")
                            app.close()
            except (KeyboardInterrupt, EOFError):
                print("\nSkipping FiftyOne app launch")
        
        print("\nAnalysis completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 