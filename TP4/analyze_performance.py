#!/usr/bin/env python3
"""
Performance Analysis Script for TP4 OpenMP Exercises
Generates plots for Exercise 4: Barrier and Synchronization Performance
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_ex4_performance():
    """Plot performance metrics for Exercise 4"""
    
    csv_file = 'ex4_clean.csv' if os.path.exists('ex4_clean.csv') else 'ex4_results.csv'
    
    if not os.path.exists(csv_file):
        print(f"{csv_file} not found. Run './ex4' first.")
        return
    
    # Read data
    df = pd.read_csv(csv_file)
    
    # Get unique versions and thread counts
    versions = df['Version'].unique()
    threads = sorted(df['Threads'].unique())
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    # Plot 1: Execution Time
    ax = axes[0, 0]
    for i, version in enumerate(versions):
        data = df[df['Version'] == version]
        ax.plot(data['Threads'], data['Time'], 
               marker=markers[i % len(markers)], linewidth=2, markersize=8,
               label=version, color=colors[i % len(colors)])
    
    ax.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Execution Time vs Threads', fontsize=14, fontweight='bold')
    ax.set_xticks(threads)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    
    # Plot 2: Speedup
    ax = axes[0, 1]
    for i, version in enumerate(versions):
        data = df[df['Version'] == version]
        ax.plot(data['Threads'], data['Speedup'], 
               marker=markers[i % len(markers)], linewidth=2, markersize=8,
               label=version, color=colors[i % len(colors)])
    
    # Plot ideal speedup
    ax.plot(threads, threads, 'k--', linewidth=2, label='Ideal', alpha=0.5)
    
    ax.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax.set_title('Speedup vs Threads', fontsize=14, fontweight='bold')
    ax.set_xticks(threads)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 3: Efficiency
    ax = axes[1, 0]
    for i, version in enumerate(versions):
        data = df[df['Version'] == version]
        ax.plot(data['Threads'], data['Efficiency'], 
               marker=markers[i % len(markers)], linewidth=2, markersize=8,
               label=version, color=colors[i % len(colors)])
    
    ax.axhline(y=100, color='k', linestyle='--', linewidth=2, alpha=0.5, label='Ideal')
    ax.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
    ax.set_ylabel('Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_title('Parallel Efficiency vs Threads', fontsize=14, fontweight='bold')
    ax.set_xticks(threads)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 120)
    
    # Plot 4: MFLOPS
    ax = axes[1, 1]
    for i, version in enumerate(versions):
        data = df[df['Version'] == version]
        ax.plot(data['Threads'], data['MFLOPS'], 
               marker=markers[i % len(markers)], linewidth=2, markersize=8,
               label=version, color=colors[i % len(colors)])
    
    ax.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (MFLOP/s)', fontsize=12, fontweight='bold')
    ax.set_title('Computational Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(threads)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('ex4_performance.png', dpi=300, bbox_inches='tight')
    print("Plot saved to ex4_performance.png")
    
    # Print performance table
    print("\n" + "="*100)
    print("EXERCISE 4: BARRIER AND SYNCHRONIZATION PERFORMANCE ANALYSIS")
    print("="*100)
    
    for version in versions:
        print(f"\n{version}:")
        print("-" * 100)
        print(f"{'Threads':>8} | {'Time (s)':>12} | {'Speedup':>10} | {'Efficiency (%)':>15} | {'MFLOP/s':>12}")
        print("-" * 100)
        
        data = df[df['Version'] == version]
        for _, row in data.iterrows():
            print(f"{int(row['Threads']):>8} | {row['Time']:>12.6f} | {row['Speedup']:>10.2f} | "
                  f"{row['Efficiency']:>15.2f} | {row['MFLOPS']:>12.2f}")
    
    print("="*100)
    

def main():
    plot_ex4_performance()

if __name__ == '__main__':
    main()
