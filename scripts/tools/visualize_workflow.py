#!/usr/bin/env python3
"""
Workflow Visualization Tool

This script generates a visual representation of the workflow between
counting and measurement components, showing how they interact in the
overall pipeline. The output can be used in documentation or presentations.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from pathlib import Path

# Define the workflow components and their relationships
WORKFLOW_COMPONENTS = {
    "Image Collection": {
        "type": "data",
        "next": ["Preprocessing"]
    },
    "Preprocessing": {
        "type": "data",
        "next": ["Object Detection", "Image Segmentation"]
    },
    "Object Detection": {
        "type": "counting",
        "next": ["Object Tracking", "Measurement"]
    },
    "Object Tracking": {
        "type": "counting",
        "next": ["Counting"]
    },
    "Counting": {
        "type": "counting",
        "next": ["Analysis"]
    },
    "Image Segmentation": {
        "type": "measurement",
        "next": ["Measurement"]
    },
    "Measurement": {
        "type": "measurement",
        "next": ["Size Analysis"]
    },
    "Size Analysis": {
        "type": "measurement",
        "next": ["Analysis"]
    },
    "Analysis": {
        "type": "output",
        "next": []
    }
}

# Define components and their implementations in the codebase
COMPONENT_IMPLEMENTATIONS = {
    "Image Collection": [
        "scripts/preprocessing/organize_test_data.py"
    ],
    "Preprocessing": [
        "src/measurement/image_enhancement/",
        "src/measurement/calibration/undistortion_script.py"
    ],
    "Object Detection": [
        "src/counting/detection/",
        "scripts/deployment/pred.py"
    ],
    "Object Tracking": [
        "src/counting/tracking/"
    ],
    "Counting": [
        "src/counting/"
    ],
    "Image Segmentation": [
        "src/measurement/segmentation/",
        "scripts/preprocessing/segment_molt.py"
    ],
    "Measurement": [
        "src/measurement/keypoints/",
        "src/measurement/measurements_calculator.py"
    ],
    "Size Analysis": [
        "scripts/analysis/analyze_sizes.py",
        "scripts/analysis/analyze_sizes_by_group.py"
    ],
    "Analysis": [
        "scripts/visualization/visualize_predictions.py",
        "scripts/visualization/show_examples.py",
        "notebooks/analysis/"
    ]
}

# Color mapping for component types
COLOR_MAP = {
    "data": "#B3E5FC",       # Light blue
    "counting": "#FFCCBC",   # Light orange/coral
    "measurement": "#C8E6C9", # Light green
    "output": "#E1BEE7"      # Light purple
}

def create_workflow_diagram(output_path=None):
    """Create a visual diagram of the workflow."""
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for component, info in WORKFLOW_COMPONENTS.items():
        G.add_node(component, type=info["type"])
    
    # Add edges
    for component, info in WORKFLOW_COMPONENTS.items():
        for next_component in info["next"]:
            G.add_edge(component, next_component)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Set up positions using hierarchical layout
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    
    # Draw nodes with colors based on component type
    for type_name, color in COLOR_MAP.items():
        nodes = [n for n, data in G.nodes(data=True) if data.get("type") == type_name]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, 
                               node_size=3000, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Create legend
    legend_elements = [
        patches.Patch(facecolor=COLOR_MAP["data"], edgecolor='black', label='Data Processing'),
        patches.Patch(facecolor=COLOR_MAP["counting"], edgecolor='black', label='Counting Algorithms'),
        patches.Patch(facecolor=COLOR_MAP["measurement"], edgecolor='black', label='Measurement Algorithms'),
        patches.Patch(facecolor=COLOR_MAP["output"], edgecolor='black', label='Output & Analysis')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add title
    plt.title('Counting and Measurement Workflow', fontsize=16, fontweight='bold')
    
    # Turn off axis
    plt.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Workflow diagram saved to {output_path}")
    else:
        plt.show()

def create_implementation_text(output_path=None):
    """Create a text file showing component implementations in codebase."""
    lines = ["# Workflow Component Implementations", ""]
    
    for component, implementations in COMPONENT_IMPLEMENTATIONS.items():
        comp_type = WORKFLOW_COMPONENTS[component]["type"]
        lines.append(f"## {component} ({comp_type.capitalize()})")
        lines.append("")
        
        if implementations:
            for impl in implementations:
                lines.append(f"- `{impl}`")
        else:
            lines.append("- *No specific implementation found*")
            
        lines.append("")
    
    content = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Implementation documentation saved to {output_path}")
    else:
        print(content)

def main():
    """Main function to generate workflow visualizations."""
    parser = argparse.ArgumentParser(description='Visualize workflow between counting and measurement components')
    parser.add_argument('--diagram', type=str, default=None, help='Output path for workflow diagram')
    parser.add_argument('--text', type=str, default=None, help='Output path for implementation text')
    parser.add_argument('--all', action='store_true', help='Generate all outputs with default names')
    args = parser.parse_args()
    
    # Determine if running from scripts directory or root
    is_in_scripts_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) == "scripts"
    
    # Get base directory (repository root)
    if is_in_scripts_dir:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(base_dir, "results", "figures")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    docs_dir = os.path.join(base_dir, "docs", "workflow")
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
    
    # Generate outputs
    if args.all or args.diagram:
        diagram_path = args.diagram if args.diagram else os.path.join(results_dir, "workflow_diagram.png")
        create_workflow_diagram(diagram_path)
    
    if args.all or args.text:
        text_path = args.text if args.text else os.path.join(docs_dir, "component_implementations.md")
        create_implementation_text(text_path)
    
    # If no output specified, show interactive diagram
    if not (args.all or args.diagram or args.text):
        create_workflow_diagram()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 