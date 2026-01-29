#!/bin/bash

# Matrix sizes and block sizes
Ns=(1000 5000 10000 20000)
NBs=(1 2 4 8 16 32 64 128 256)

# Output file
OUT="hpl_results.txt"

echo "N NB Time(s) GFLOPS Status" > $OUT

for N in "${Ns[@]}"; do
  for NB in "${NBs[@]}"; do

    echo "Running HPL: N=$N NB=$NB"

    # Update HPL.dat
    # Update N (line 6)
    sed -i "6s/.*/$N         Ns/" HPL.dat

    # Update NB (line 8)
    sed -i "8s/.*/$NB         NBs/" HPL.dat

    # Run HPL
    srun --mpi=pmix -n 1 ./xhpl > hpl.out

    # Extract results
    TIME=$(grep WR00C2R4 hpl.out | awk '{print $(NF-1)}')
    GFLOPS=$(grep WR00C2R4 hpl.out | awk '{print $NF}')
    STATUS=$(grep WR00C2R4 hpl.out | awk '{print $(NF-2)}')

    echo "$N $NB $TIME $GFLOPS $STATUS" | tee -a $OUT

  done
done
