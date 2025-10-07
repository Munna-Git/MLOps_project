#!/bin/bash
# entrypoint-wrapper.sh


# Run the training script first
python app/train_entrypoint.py

# Once training completes, run the inference script
python app/inference_entrypoint.py
