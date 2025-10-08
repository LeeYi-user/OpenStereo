#!/usr/bin/env python3
"""
Plot training loss from log file
Usage: python plot_training_loss.py <log_file_path>
"""

import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_log_file(log_path):
    """
    Parse training log file and extract iteration and loss information
    
    Args:
        log_path: Path to the log file
        
    Returns:
        iterations: List of iteration numbers
        current_losses: List of current batch losses
        avg_losses: List of average losses
    """
    iterations = []
    current_losses = []
    avg_losses = []
    
    # Regular expression to match training log lines
    # Example: Training Epoch: 0/1 Iter:   0/26612 Loss:17.2630(17.2630)
    pattern = r'Training Epoch:.*?Iter:\s+(\d+)/\d+\s+Loss:([\d.]+)\(([\d.]+)\)'
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                iteration = int(match.group(1))
                current_loss = float(match.group(2))
                avg_loss = float(match.group(3))
                
                iterations.append(iteration)
                current_losses.append(current_loss)
                avg_losses.append(avg_loss)
    
    return iterations, current_losses, avg_losses


def plot_losses(iterations, current_losses, avg_losses, output_path=None):
    """
    Plot training losses
    
    Args:
        iterations: List of iteration numbers
        current_losses: List of current batch losses
        avg_losses: List of average losses
        output_path: Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Both current and average losses
    ax1.plot(iterations, current_losses, 'b-', alpha=0.3, label='Current Batch Loss', linewidth=0.5)
    ax1.plot(iterations, avg_losses, 'r-', label='Average Loss', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss vs Iteration', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average loss only (clearer view)
    ax2.plot(iterations, avg_losses, 'r-', label='Average Loss', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Average Loss', fontsize=12)
    ax2.set_title('Average Training Loss vs Iteration', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def print_statistics(iterations, current_losses, avg_losses):
    """Print training statistics"""
    print("\n" + "="*60)
    print("Training Statistics")
    print("="*60)
    print(f"Total iterations logged: {len(iterations)}")
    print(f"Iteration range: {iterations[0]} - {iterations[-1]}")
    print(f"\nCurrent Batch Loss:")
    print(f"  Min: {np.min(current_losses):.4f}")
    print(f"  Max: {np.max(current_losses):.4f}")
    print(f"  Mean: {np.mean(current_losses):.4f}")
    print(f"  Std: {np.std(current_losses):.4f}")
    print(f"\nAverage Loss:")
    print(f"  Initial: {avg_losses[0]:.4f}")
    print(f"  Final: {avg_losses[-1]:.4f}")
    print(f"  Min: {np.min(avg_losses):.4f}")
    print(f"  Max: {np.max(avg_losses):.4f}")
    print(f"  Improvement: {avg_losses[0] - avg_losses[-1]:.4f} ({(avg_losses[0] - avg_losses[-1])/avg_losses[0]*100:.2f}%)")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot training loss from log file')
    parser.add_argument('log_file', type=str, help='Path to the training log file')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='Output path for the plot image (e.g., loss_plot.png)')
    parser.add_argument('--no-show', action='store_true', 
                        help='Do not display the plot window')
    
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return
    
    print(f"Parsing log file: {log_path}")
    iterations, current_losses, avg_losses = parse_log_file(log_path)
    
    if not iterations:
        print("Error: No training data found in log file")
        return
    
    print(f"Found {len(iterations)} training iterations")
    
    # Print statistics
    print_statistics(iterations, current_losses, avg_losses)
    
    # Set default output path if not specified
    output_path = args.output
    if output_path is None and not args.no_show:
        output_dir = log_path.parent
        output_path = output_dir / f"{log_path.stem}_loss_plot.png"
    
    # Plot losses
    if args.no_show:
        if output_path:
            plt.ioff()
            plot_losses(iterations, current_losses, avg_losses, output_path)
            plt.close()
        else:
            print("Warning: --no-show specified but no output path given")
    else:
        plot_losses(iterations, current_losses, avg_losses, output_path)


if __name__ == '__main__':
    main()
