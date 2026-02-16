#!/usr/bin/env python3
"""
Performance Analysis Script for TP3 OpenMP Exercises
Generates speedup and efficiency plots for exercises 4 and 5
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def calculate_speedup_efficiency(csv_file, time_column='time'):
    """Calculate speedup and efficiency from timing data"""
    df = pd.read_csv(csv_file)
    
    # Get baseline (serial) time
    t1 = df[df['threads'] == 1][time_column].values[0]
    
    # Calculate speedup and efficiency
    df['speedup'] = t1 / df[time_column]
    df['efficiency'] = df['speedup'] / df['threads'] * 100
    
    return df

def plot_performance(df, title, output_file):
    """Create speedup and efficiency plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    threads = df['threads']
    speedup = df['speedup']
    efficiency = df['efficiency']
    
    # Speedup plot
    ax1.plot(threads, speedup, 'bo-', linewidth=2, markersize=8, label='Actual Speedup')
    ax1.plot(threads, threads, 'r--', linewidth=2, label='Ideal Speedup')
    ax1.set_xlabel('Number of Threads', fontsize=12)
    ax1.set_ylabel('Speedup', fontsize=12)
    ax1.set_title(f'{title} - Speedup', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xticks(threads)
    
    # Efficiency plot
    ax2.plot(threads, efficiency, 'go-', linewidth=2, markersize=8, label='Efficiency')
    ax2.axhline(y=100, color='r', linestyle='--', linewidth=2, label='Ideal Efficiency')
    ax2.set_xlabel('Number of Threads', fontsize=12)
    ax2.set_ylabel('Efficiency (%)', fontsize=12)
    ax2.set_title(f'{title} - Efficiency', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xticks(threads)
    ax2.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()

def print_performance_table(df, title):
    """Print formatted performance table"""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    print(f"{'Threads':>8} | {'Time (s)':>12} | {'Speedup':>10} | {'Efficiency (%)':>15}")
    print(f"{'-'*70}")
    
    for _, row in df.iterrows():
        print(f"{int(row['threads']):>8} | {row['time']:>12.6f} | {row['speedup']:>10.2f} | {row['efficiency']:>15.2f}")
    
    print(f"{'='*70}\n")

def main():
    # Exercise 4 - Matrix Multiplication
    if os.path.exists('ex4_results.csv'):
        print("\nAnalyzing Exercise 4 (Matrix Multiplication)...")
        df_ex4 = calculate_speedup_efficiency('ex4_results.csv')
        print_performance_table(df_ex4, 'Exercise 4: Matrix Multiplication')
        plot_performance(df_ex4, 'Exercise 4: Matrix Multiplication', 'ex4_performance.png')
    else:
        print("ex4_results.csv not found. Run 'make perf-ex4' first.")
    
    # Exercise 5 - Jacobi Method
    if os.path.exists('ex5_results.csv'):
        print("\nAnalyzing Exercise 5 (Jacobi Method)...")
        df_ex5 = calculate_speedup_efficiency('ex5_results.csv')
        print_performance_table(df_ex5, 'Exercise 5: Jacobi Method')
        plot_performance(df_ex5, 'Exercise 5: Jacobi Method', 'ex5_performance.png')
    else:
        print("ex5_results.csv not found. Run 'make perf-ex5' first.")

if __name__ == '__main__':
    main()
