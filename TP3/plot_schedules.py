#!/usr/bin/env python3
"""
Plot scheduling policy comparison for Exercise 4
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_schedules():
    # Read data
    df = pd.read_csv('ex4_schedules_results.csv')
    
    # Get unique values
    thread_counts = sorted(df['threads'].unique())
    schedules = df['schedule'].unique()
    
    # Create subplots for each thread count
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, threads in enumerate(thread_counts):
        ax = axes[idx]
        df_threads = df[df['threads'] == threads]
        
        for schedule in schedules:
            df_schedule = df_threads[df_threads['schedule'] == schedule]
            df_schedule = df_schedule.sort_values('chunk_size')
            
            # Replace 0 chunk_size with 'default' for better visualization
            labels = ['default' if cs == 0 else str(cs) for cs in df_schedule['chunk_size']]
            
            ax.plot(range(len(df_schedule)), df_schedule['time'], 
                   marker='o', linewidth=2, markersize=6, label=schedule.upper())
        
        ax.set_xlabel('Chunk Size', fontsize=10)
        ax.set_ylabel('Execution Time (s)', fontsize=10)
        ax.set_title(f'Threads: {threads}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Set x-axis labels
        chunk_sizes = sorted(df_threads['chunk_size'].unique())
        labels = ['default' if cs == 0 else str(cs) for cs in chunk_sizes]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
    
    # Hide extra subplot if we have less than 6 thread counts
    for idx in range(len(thread_counts), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Matrix Multiplication: Schedule Policy Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex4_schedules_comparison.png', dpi=300, bbox_inches='tight')
    print("Schedule comparison plot saved to ex4_schedules_comparison.png")
    
    # Print best configurations
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS FOR EACH THREAD COUNT")
    print("="*80)
    for threads in thread_counts:
        df_threads = df[df['threads'] == threads]
        best = df_threads.loc[df_threads['time'].idxmin()]
        chunk_str = 'default' if best['chunk_size'] == 0 else str(int(best['chunk_size']))
        print(f"Threads: {threads:2d} | Best: {best['schedule']:8s} | "
              f"Chunk: {chunk_str:7s} | Time: {best['time']:.6f}s")
    print("="*80 + "\n")

if __name__ == '__main__':
    plot_schedules()
