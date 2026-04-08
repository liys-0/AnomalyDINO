#!/bin/bash
# Loop over parameters
#for SHOTS in 1 2 4 8 20 40 60 80 100; do   
for SHOTS in 1 2 4 8 20 40 60 80; do   
    # Submit the job and pass the variables to the SLURM script
    sbatch --export=ALL,SHOTS=$SHOTS run_base.sh
    
    # Optional: sleep for a second so you don't overwhelm the scheduler
    sleep 1
done