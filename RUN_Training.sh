#!/bin/bash

# Define the number of GPUs available
NUM_GPUS=4
MAX_RUN_ID=5

# Define the script or command to run for training
TRAINING_SCRIPT="/conda/ozavala/miniconda/envs/aieoas/bin/python /unity/f1/ozavala/CODE/da_hycom/3_Train_2D.py"

# Directory to store logs
LOG_DIR="/unity/f1/ozavala/OUTPUTS/DA_HYCOM_TSIS/run_logs/"
mkdir -p $LOG_DIR  # Create the log directory if it doesn't exist

# Loop over RUN_ID from 1 to 5
for RUN_ID in $(seq 1 $MAX_RUN_ID); do
  echo "Starting RUN_ID: $RUN_ID"
  
  # Loop over the number of processes
  for ((i=0; i<$NUM_GPUS; i++)); do
    # Calculate which GPU to use for this process
    GPU_ID=$((i % NUM_GPUS))
    
    # Define the log file for this process
    LOG_FILE="$LOG_DIR/train_run_${RUN_ID}_gpu_$((GPU_ID + 1)).log"
    
    # Prepare the full command
    FULL_COMMAND="CUDA_VISIBLE_DEVICES=$GPU_ID $TRAINING_SCRIPT $((GPU_ID + 1)) $RUN_ID > $LOG_FILE 2>&1 &"
    
    # Echo the full command to the console
    echo "Executing: $FULL_COMMAND"
    
    # Execute the command
    eval $FULL_COMMAND
    
    # Optionally, introduce a delay to stagger the starts (helps avoid conflicts)
    sleep 1
  done
  
  # Wait for all background processes of this RUN_ID to finish before starting the next RUN_ID
  wait
  
  echo "Completed RUN_ID: $RUN_ID"
done

echo "All training processes for all RUN_IDs completed."