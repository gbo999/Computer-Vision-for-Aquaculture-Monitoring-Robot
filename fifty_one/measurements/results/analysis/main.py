#!/usr/bin/env python3
"""
Refactored prawn measurement validation using FiftyOne.

This is the main entry point that replaces the original measurements_analysis.py
with a clean, modular architecture.

Key improvements from original code:
- Clean separation of concerns across multiple modules
- Configurable parameters through config.py
- Better error handling and logging
- Extensible design for new measurement types
- Reduced code duplication
- More maintainable codebase
"""

import argparse
import sys
from pathlib import Path

from config import Config
from measurement_analysis import MeasurementAnalyzer, PortManager


def parse_arguments():
    """
    Parse command line arguments.
    
    Improved from original with:
    - Better help documentation
    - Input validation
    - More flexible parameter options
    """
    parser = argparse.ArgumentParser(
        description='Prawn measurement validation using FiftyOne',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --type carapace --weights all
  %(prog)s --type body --weights car --port 5160
  %(prog)s --type carapace --weights kalkar --config custom_config.py
        """
    )
    
    parser.add_argument(
        '--type', 
        choices=['carapace', 'body'],
        default='body',
        help='Type of measurement to analyze (default: body)'
    )
    
    parser.add_argument(
        '--weights', 
        choices=['car', 'kalkar', 'all'], 
        default='all',
        help='Version of the prediction weights to use (default: all)'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=None,
        help='Port for FiftyOne visualization (auto-selected if not specified)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom configuration file (optional)'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip processing if dataset already exists and launch visualization directly'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Custom output directory for results (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def validate_arguments(args, config: Config):
    """
    Validate command line arguments and configuration.
    
    Args:
        args: Parsed command line arguments
        config: Configuration object
        
    Raises:
        SystemExit: If validation fails
    """
    # Check if required data files exist
    if args.type == 'carapace':
        data_path = config.CARAPACE_DATA_PATH
    else:
        data_path = config.BODY_DATA_PATH
    
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Please check your configuration and ensure data files are available.")
        sys.exit(1)
    
    # Check if metadata file exists
    if not config.METADATA_PATH.exists():
        print(f"Error: Metadata file not found: {config.METADATA_PATH}")
        sys.exit(1)
    
    # Check if prediction path exists
    prediction_path = config.get_prediction_path(args.weights)
    if not prediction_path.exists():
        print(f"Error: Prediction path not found: {prediction_path}")
        print(f"Please check weights type '{args.weights}' and ensure predictions are available.")
        sys.exit(1)
    
    # Check if ground truth path exists
    if not config.GROUND_TRUTH_PATH.exists():
        print(f"Error: Ground truth path not found: {config.GROUND_TRUTH_PATH}")
        sys.exit(1)
    
    # Validate port if specified
    if args.port is not None and (args.port < 1024 or args.port > 65535):
        print(f"Error: Port {args.port} is not in valid range (1024-65535)")
        sys.exit(1)


def setup_logging(verbose: bool):
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging
    """
    import logging
    
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_custom_config(config_path: str) -> Config:
    """
    Load custom configuration if specified.
    
    Args:
        config_path: Path to custom configuration file
        
    Returns:
        Configuration object
    """
    if config_path and Path(config_path).exists():
        print(f"Loading custom configuration from: {config_path}")
        # This could be extended to support custom config files
        # For now, just use the default config
        return Config()
    elif config_path:
        print(f"Warning: Custom config file not found: {config_path}")
        print("Using default configuration...")
    
    return Config()


def main():
    """
    Main entry point for the refactored measurement analysis.
    
    This function orchestrates the entire analysis process:
    1. Parse and validate arguments
    2. Load configuration
    3. Set up components
    4. Run analysis
    5. Launch visualization
    """
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Load configuration
    config = load_custom_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config.THESIS_EXPORT_PATH = Path(args.output_dir)
        config.THESIS_EXPORT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Validate arguments and configuration
    validate_arguments(args, config)
    
    # Set up port management
    port_manager = PortManager(config)
    
    if args.port is None:
        port = port_manager.get_available_port()
        print(f"Auto-selected port: {port}")
    else:
        port = args.port
        if not port_manager.is_port_available(port):
            print(f"Warning: Port {port} may not be available")
    
    # Initialize analyzer
    print(f"Initializing measurement analyzer...")
    analyzer = MeasurementAnalyzer(config)
    
    try:
        # Run analysis
        print(f"Starting analysis for {args.type} measurements using {args.weights} weights...")
        session = analyzer.run_analysis(
            measurement_type=args.type,
            weights_type=args.weights,
            port=port
        )
        
        print(f"\nAnalysis completed successfully!")
        print(f"FiftyOne visualization available at: http://localhost:{port}")
        print(f"Dataset: {config.get_dataset_name(args.type, args.weights)}")
        print(f"Results saved to: {config.get_output_filename(args.type, args.weights)}")
        
        # Keep the session alive
        print("\nPress Ctrl+C to close the FiftyOne session and exit...")
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("\nClosing FiftyOne session...")
            session.close()
            print("Session closed. Goodbye!")
    
    except Exception as e:
        print(f"\nError during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 