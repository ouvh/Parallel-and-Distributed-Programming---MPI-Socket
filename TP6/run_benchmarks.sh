#!/bin/bash
# Shell script to run performance benchmarks for Exercise 2

echo "Running Exercise 2 (Gradient Descent) performance tests..."
echo "Processes,Samples,Time,FinalLoss,Epochs" > ex2_performance.csv

# Test different sample sizes and process counts
for samples in 5000 10000 20000
do
    echo "Testing with $samples samples..."
    
    # Serial (1 process)
    output=$(mpirun -np 1 ./ex2 $samples 2>&1)
    time=$(echo "$output" | grep "Training time:" | awk '{print $3}')
    loss=$(echo "$output" | grep "Final loss:" | awk '{print $3}')
    epochs=$(echo "$output" | grep -E "(Early stopping|MAX_EPOCHS)" | awk '{print $4}' | tr -d ',')
    
    if [ -z "$epochs" ]; then
        epochs=$(echo "$output" | tail -20 | grep "Epoch" | tail -1 | awk '{print $2}')
    fi
    
    echo "1,$samples,$time,$loss,$epochs" >> ex2_performance.csv
    
    # Parallel with different process counts
    for np in 2 4 8 16 32
    do
        output=$(mpirun -np $np ./ex2 $samples 2>&1)
        time=$(echo "$output" | grep "Training time:" | awk '{print $3}')
        loss=$(echo "$output" | grep "Final loss:" | awk '{print $3}')
        epochs=$(echo "$output" | grep -E "(Early stopping|MAX_EPOCHS)" | awk '{print $4}' | tr -d ',')
        
        if [ -z "$epochs" ]; then
            epochs=$(echo "$output" | tail -20 | grep "Epoch" | tail -1 | awk '{print $2}')
        fi
        
        echo "$np,$samples,$time,$loss,$epochs" >> ex2_performance.csv
        sleep 1
    done
done

echo "Performance tests complete! Results in ex2_performance.csv"
