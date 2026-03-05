"""
Performance Analysis Script for TP6 MPI Derived Types
Generates plots for Exercise 2 (Distributed Gradient Descent)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_ex2():
    """Analyze Exercise 2: Distributed Gradient Descent"""
    try:
        df = pd.read_csv('ex2_performance.csv')
    except FileNotFoundError:
        print("ex2_performance.csv not found. Please run benchmarks first.")
        return
    
    print("=" * 70)
    print("Exercise 2: Distributed Gradient Descent Performance")
    print("=" * 70)
    
    # Get unique sample sizes
    sample_sizes = df['Samples'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Exercise 2: MPI Distributed Gradient Descent Performance', 
                 fontsize=14, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, samples in enumerate(sample_sizes):
        df_samples = df[df['Samples'] == samples]
        
        # Get serial time
        serial_time = df_samples[df_samples['Processes'] == 1]['Time'].values[0]
        
        # Calculate metrics
        processes = df_samples['Processes'].values
        times = df_samples['Time'].values
        speedup = serial_time / times
        efficiency = (speedup / processes) * 100
        
        color = colors[idx % len(colors)]
        label = f'{samples} samples'
        
        # Plot 1: Execution Time
        axes[0, 0].plot(processes, times, 'o-', label=label, color=color, linewidth=2)
        
        # Plot 2: Speedup
        axes[0, 1].plot(processes, speedup, 'o-', label=label, color=color, linewidth=2)
        axes[0, 1].plot(processes, processes, '--', color='gray', alpha=0.5,
                       label='Ideal' if idx == 0 else '')
        
        # Plot 3: Efficiency
        axes[1, 0].plot(processes, efficiency, 'o-', label=label, color=color, linewidth=2)
        axes[1, 0].axhline(y=100, color='gray', linestyle='--', alpha=0.5,
                          label='Ideal' if idx == 0 else '')
        
        # Print table
        print(f"\n--- Sample Size: {samples} ---")
        print(f"{'Processes':<12} {'Time (s)':<12} {'Speedup':<12} {'Efficiency (%)':<15}")
        print("-" * 70)
        for i, (p, t, s, e) in enumerate(zip(processes, times, speedup, efficiency)):
            print(f"{p:<12} {t:<12.6f} {s:<12.2f} {e:<15.2f}")
    
    # Plot 4: Training Rate (samples processed per second)
    df_best = df[df['Samples'] == sample_sizes[-1]]
    processes = df_best['Processes'].values
    times = df_best['Time'].values
    epochs = df_best['Epochs'].values
    
    # Approximate iterations per second
    throughput = (sample_sizes[-1] * epochs) / times / 1000  # K samples*epochs/sec
    
    axes[1, 1].plot(processes, throughput, 'o-', color='#d62728', linewidth=2)
    axes[1, 1].set_xlabel('Number of Processes', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Throughput (K samples×epochs/s)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Training Throughput', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Configure subplots
    axes[0, 0].set_xlabel('Number of Processes', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Training Time', fontsize=12, fontweight='bold')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Number of Processes', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Speedup', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Speedup vs Ideal', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Number of Processes', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Efficiency (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Parallel Efficiency', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ex2_performance.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to ex2_performance.png")

def main():
    analyze_ex2()
    print("\n" + "=" * 70)
    print("Performance analysis complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()
