#!/bin/bash

# Define the number of GPUs available
NUM_GPUS=4

# Define the number of training processes you want to run
NUM_PROCESSES=4

# Define the script or command to run for training
TRAINING_SCRIPT=" /conda/ozavala/miniconda/envs/aieoas/bin/python /unity/f1/ozavala/CODE/da_hycom/3_Train_2D.py"  # Replace with your actual script

# Directory to store logs
LOG_DIR="/unity/f1/ozavala/OUTPUTS/DA_HYCOM_TSIS/run_logs/"
mkdir -p $LOG_DIR  # Create the log directory if it doesn't exist

# Loop over the number of processes
for ((i=0; i<$NUM_PROCESSES; i++)); do
  # Calculate which GPU to use for this process
  GPU_ID=$((i % NUM_GPUS))
  
  # Define the log file for this process
  LOG_FILE="$LOG_DIR/train_gpu_$((GPU_ID + 1)).log"
  # Export the CUDA_VISIBLE_DEVICES environment variable for this process

   # Prepare the full command
  FULL_COMMAND="CUDA_VISIBLE_DEVICES=$GPU_ID $TRAINING_SCRIPT $((GPU_ID + 1)) > $LOG_FILE 2>&1 &"
  
  # Echo the full command to the console
  echo "Executing: $FULL_COMMAND"
  
  # Execute the command
  eval $FULL_COMMAND
  
  # Optionally, introduce a delay to stagger the starts (helps avoid conflicts)
  sleep 1
done

# Wait for all background processes to finish
wait

echo "All training processes completed."
