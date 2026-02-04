import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

# ================= EXERCISE 1 =================
try:
    df_o0_float = pd.read_csv('ex1_O0_float.csv')
    df_o0_int = pd.read_csv('ex1_O0_int.csv')
    
    # Read O2 single points
    df_o2_float = pd.read_csv('ex1_O2_float.csv')
    df_o2_int = pd.read_csv('ex1_O2_int.csv')
    
    val_o2_float = df_o2_float['Time ms'].iloc[0] if not df_o2_float.empty else None
    val_o2_int = df_o2_int['Time ms'].iloc[0] if not df_o2_int.empty else None

    plt.figure(figsize=(10, 6))
    plt.plot(df_o0_float['unrolling factor'], df_o0_float['Time ms'], 'o-', label='O0 Float')
    plt.plot(df_o0_int['unrolling factor'], df_o0_int['Time ms'], 's-', label='O0 Int')
    
    if val_o2_float:
        plt.axhline(y=val_o2_float, color='r', linestyle='--', label=f'O2 Float (Base): {val_o2_float:.2f}ms')
    if val_o2_int:
        plt.axhline(y=val_o2_int, color='purple', linestyle='--', label=f'O2 Int (Base): {val_o2_int:.2f}ms')

    plt.xlabel('Unrolling Factor')
    plt.ylabel('Execution Time (ms)')
    plt.title('Exercise 1: Loop Unrolling Impact (O0 vs O2)')
    plt.legend()
    plt.xscale('log', base=2)
    plt.grid(True)
    plt.savefig('ex1_tp2_plot.png')
    print("Generated ex1_tp2_plot.png")
except Exception as e:
    print(f"Skipping Ex1 plot: {e}")

# ================= EXERCISE 3 & 4 (Theoretical Speedup) =================
def plot_speedup_curve(fs_list, labels, title, filename, show_linear=True, y_limit=None):
    p = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    plt.figure(figsize=(10, 6))
    
    for fs, label in zip(fs_list, labels):
        # Amdahl's Law: S = 1 / (fs + (1-fs)/p)
        amdahl = 1 / (fs + (1 - fs) / p)
        plt.plot(p, amdahl, 'o-', label=f"{label} (fs={fs:.1%})")
    
    # Ideal Linear Reference
    if show_linear:
        plt.plot(p, p, 'k--', alpha=0.3, label="Ideal Linear")
    
    plt.xlabel('Number of Processors (p)')
    plt.ylabel('Speedup S(p) (Amdahl)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if y_limit:
        plt.ylim(y_limit)
    else:
        plt.ylim(bottom=1)
        
    plt.savefig(filename)
    print(f"Generated {filename}")

# Exercise 3: Constant Fraction over N
# Data from ex3_varying_n.txt shows ~46% for all N
fs_ex3 = [0.46]
# Limit Y axis to 4 to show the curve saturation clearly
plot_speedup_curve(fs_ex3, ['All N sizes'], 
                   'Exercise 3: Amdahl Speedup (Constant fs)', 
                   'ex3_tp2_plot.png', 
                   show_linear=False,  # Hide linear line as it distorts scale for this specific plot
                   y_limit=(1, 3))

# Exercise 4: Varying Fraction over N
# Data from ex4_varying_n.txt:
# N=20  -> fs ~ 36.6% (dominated by overhead)
# N=100 -> fs ~ 2.4%
# N=512 -> fs ~ 0.35%
fs_ex4 = [0.366, 0.024, 0.0035]
labels_ex4 = ['N=20', 'N=100', 'N=512']
plot_speedup_curve(fs_ex4, labels_ex4, 'Exercise 4: Amdahl Speedup vs Problem Size (N)', 'ex4_tp2_plot.png', show_linear=True)

# ================= EXERCISE 3 & 4 (Weak Scaling / Gustafson) =================
def plot_weak_scaling_curve(fs_list, labels, title, filename):
    p = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    plt.figure(figsize=(10, 6))
    
    for fs, label in zip(fs_list, labels):
        # Gustafson's Law: S = fs + p*(1-fs)
        gustafson = fs + p * (1 - fs)
        plt.plot(p, gustafson, 'o-', label=f"{label} (fs={fs:.1%})")
    
    # Ideal Linear Reference
    plt.plot(p, p, 'k--', alpha=0.3, label="Ideal Linear")
    
    plt.xlabel('Number of Processors (p)')
    plt.ylabel('Scaled Speedup S(p) (Gustafson)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=1)
        
    plt.savefig(filename)
    print(f"Generated {filename}")

# Exercise 3 Weak Scaling
plot_weak_scaling_curve(fs_ex3, ['All N sizes'], 
                        'Exercise 3: Gustafson Speedup (Weak Scaling)', 
                        'ex3_tp2_weak_plot.png')

# Exercise 4 Weak Scaling
plot_weak_scaling_curve(fs_ex4, labels_ex4, 
                        'Exercise 4: Gustafson Speedup (Weak Scaling)', 
                        'ex4_tp2_weak_plot.png')