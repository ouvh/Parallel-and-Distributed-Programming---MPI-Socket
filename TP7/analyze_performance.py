#!/usr/bin/env python3
"""
Performance analysis script for TP7 - Poisson Solver
Generates plots for speedup, efficiency, and execution time
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def load_data(filename='ex2_performance.csv'):
    """Load performance data from CSV file"""
    try:
        df = pd.read_csv(filename)
        # Filter out rows with N/A or invalid data
        df = df[df['Time_s'] != 'N/A']
        df['Time_s'] = pd.to_numeric(df['Time_s'], errors='coerce')
        df = df.dropna(subset=['Time_s'])
        return df
    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        print("Please run the benchmark script first: bash run_benchmarks.sh")
        sys.exit(1)

def compute_metrics(df):
    """Compute speedup and efficiency metrics"""
    results = []
    
    for grid_size in df['Grid_Size'].unique():
        grid_data = df[df['Grid_Size'] == grid_size].copy()
        grid_data = grid_data.sort_values('Processes')
        
        # Get serial time (1 process)
        serial_time = grid_data[grid_data['Processes'] == 1]['Time_s'].values
        if len(serial_time) == 0:
            continue
        serial_time = serial_time[0]
        
        for _, row in grid_data.iterrows():
            nproc = row['Processes']
            time = row['Time_s']
            
            speedup = serial_time / time if time > 0 else 0
            efficiency = (speedup / nproc) * 100 if nproc > 0 else 0
            
            results.append({
                'Grid_Size': grid_size,
                'Processes': nproc,
                'Time_s': time,
                'Speedup': speedup,
                'Efficiency': efficiency,
                'Iterations': row['Iterations']
            })
    
    return pd.DataFrame(results)

def plot_performance(df):
    """Generate performance plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TP7 - Poisson Solver Performance Analysis', fontsize=16, fontweight='bold')
    
    grid_sizes = sorted(df['Grid_Size'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(grid_sizes)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    # Plot 1: Execution Time
    ax = axes[0, 0]
    for idx, grid_size in enumerate(grid_sizes):
        data = df[df['Grid_Size'] == grid_size].sort_values('Processes')
        ax.plot(data['Processes'], data['Time_s'], 
               marker=markers[idx % len(markers)], 
               color=colors[idx], 
               linewidth=2, markersize=8,
               label=f'{grid_size}×{grid_size}')
    ax.set_xlabel('Number of Processes', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title('Execution Time vs Process Count', fontsize=13, fontweight='bold')
    ax.legend(title='Grid Size')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Speedup
    ax = axes[0, 1]
    for idx, grid_size in enumerate(grid_sizes):
        data = df[df['Grid_Size'] == grid_size].sort_values('Processes')
        ax.plot(data['Processes'], data['Speedup'], 
               marker=markers[idx % len(markers)], 
               color=colors[idx], 
               linewidth=2, markersize=8,
               label=f'{grid_size}×{grid_size}')
    # Plot ideal speedup
    max_proc = df['Processes'].max()
    ideal_procs = np.arange(1, max_proc + 1)
    ax.plot(ideal_procs, ideal_procs, 'k--', linewidth=2, label='Ideal', alpha=0.5)
    ax.set_xlabel('Number of Processes', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title('Speedup vs Process Count', fontsize=13, fontweight='bold')
    ax.legend(title='Grid Size')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Efficiency
    ax = axes[1, 0]
    for idx, grid_size in enumerate(grid_sizes):
        data = df[df['Grid_Size'] == grid_size].sort_values('Processes')
        ax.plot(data['Processes'], data['Efficiency'], 
               marker=markers[idx % len(markers)], 
               color=colors[idx], 
               linewidth=2, markersize=8,
               label=f'{grid_size}×{grid_size}')
    ax.axhline(y=100, color='k', linestyle='--', linewidth=2, label='Ideal', alpha=0.5)
    ax.set_xlabel('Number of Processes', fontsize=12)
    ax.set_ylabel('Efficiency (%)', fontsize=12)
    ax.set_title('Parallel Efficiency', fontsize=13, fontweight='bold')
    ax.legend(title='Grid Size')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 110)
    
    # Plot 4: Strong Scaling (Time per iteration)
    ax = axes[1, 1]
    for idx, grid_size in enumerate(grid_sizes):
        data = df[df['Grid_Size'] == grid_size].sort_values('Processes')
        time_per_iter = data['Time'] / data['Iterations']
        ax.plot(data['Processes'], time_per_iter * 1000,  # Convert to ms
               marker=markers[idx % len(markers)], 
               color=colors[idx], 
               linewidth=2, markersize=8,
               label=f'{grid_size}×{grid_size}')
    ax.set_xlabel('Number of Processes', fontsize=12)
    ax.set_ylabel('Time per Iteration (ms)', fontsize=12)
    ax.set_title('Iteration Performance', fontsize=13, fontweight='bold')
    ax.legend(title='Grid Size')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('ex2_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Performance plots saved to ex2_performance.png")
    
def print_performance_table(df):
    """Print formatted performance tables"""
    print("\n" + "="*80)
    print("POISSON SOLVER PERFORMANCE ANALYSIS")
    print("="*80)
    
    for grid_size in sorted(df['Grid_Size'].unique()):
        print(f"\n--- Grid Size: {grid_size}×{grid_size} ---")
        data = df[df['Grid_Size'] == grid_size].sort_values('Processes')
        
        print(f"{'Processes':<12} {'Time (s)':<12} {'Speedup':<12} {'Efficiency (%)':<15} {'Iterations':<12}")
        print("-" * 70)
        
        for _, row in data.iterrows():
            print(f"{int(row['Processes']):<12} "
                  f"{row['Time_s']:<12.6f} "
                  f"{row['Speedup']:<12.2f} "
                  f"{row['Efficiency']:<15.2f} "
                  f"{int(row['Iterations']):<12}")
    
    # Find best configurations
    print("\n" + "="*80)
    print("OPTIMAL CONFIGURATIONS")
    print("="*80)
    
    for grid_size in sorted(df['Grid_Size'].unique()):
        data = df[df['Grid_Size'] == grid_size]
        best_speedup = data.loc[data['Speedup'].idxmax()]
        best_efficiency = data[data['Processes'] > 1].loc[data[data['Processes'] > 1]['Efficiency'].idxmax()]
        
        print(f"\nGrid {grid_size}×{grid_size}:")
        print(f"  Best speedup: {best_speedup['Speedup']:.2f}× with {int(best_speedup['Processes'])} processes "
              f"({best_speedup['Efficiency']:.1f}% efficiency)")
        print(f"  Best efficiency: {best_efficiency['Efficiency']:.1f}% with {int(best_efficiency['Processes'])} processes "
              f"({best_efficiency['Speedup']:.2f}× speedup)")

def main():
    print("="*80)
    print("TP7 - Poisson Solver Performance Analysis")
    print("="*80)
    
    # Load data
    df = load_data()
    print(f"\n✓ Loaded {len(df)} performance measurements")
    
    # Compute metrics
    df_metrics = compute_metrics(df)
    print(f"✓ Computed speedup and efficiency metrics")
    
    # Print tables
    print_performance_table(df_metrics)
    
    # Generate plots
    print("\nGenerating performance plots...")
    plot_performance(df_metrics)
    
    print("\n" + "="*80)
    print("Performance analysis complete!")
    print("="*80)

if __name__ == '__main__':
    main()
