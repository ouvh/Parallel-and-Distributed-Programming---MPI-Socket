#!/bin/bash
# Benchmark script for Exercise 2: Poisson Solver
# Tests performance with different process counts and grid sizes

OUTPUT_FILE="ex2_performance.csv"

echo "=== Poisson Solver Performance Benchmark ==="
echo "Testing various grid sizes with 1, 2, 4, 8, 16 processes"
echo ""

# Create CSV header
echo "Processes,Grid_Size,Time_s,Iterations" > $OUTPUT_FILE

# Grid sizes to test
GRID_SIZES=(64 128 256 512)
PROCESS_COUNTS=(1 2 4 8 16)

for grid_size in "${GRID_SIZES[@]}"; do
    echo "Testing grid size: ${grid_size}x${grid_size}"
    
    for nproc in "${PROCESS_COUNTS[@]}"; do
        echo "  Running with $nproc processes..."
        
        # Run the program and capture output
        output=$(mpirun -np $nproc ./ex2 $grid_size $grid_size 2>&1)
        
        # Extract timing and iteration information
        # Expected format: "Converged after X iterations in Y seconds"
        iterations=$(echo "$output" | grep "Converged after" | awk '{print $3}')
        time=$(echo "$output" | grep "Converged after" | awk '{print $6}')
        
        # If extraction failed, use default values
        if [ -z "$time" ]; then
            time="N/A"
        fi
        if [ -z "$iterations" ]; then
            iterations="N/A"
        fi
        
        echo "$nproc,$grid_size,$time,$iterations" >> $OUTPUT_FILE
        echo "    Time: $time s, Iterations: $iterations"
    done
    echo ""
done

echo "Benchmark complete! Results saved to $OUTPUT_FILE"
echo ""
echo "To analyze results, run:"
echo "  python3 analyze_performance.py"
