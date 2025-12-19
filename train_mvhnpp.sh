#!/bin/bash

# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Error: No config directory provided."
    echo "Usage: ./run_gen.sh <subject name>"
    exit 1
fi

SUBJECT_NAME="$1"

echo "Generating canonical LBS weight volume ..."
python -m gen_data.gen_weight_volume -c "configs/${SUBJECT_NAME}/template.yaml"

echo "Reconstructing a template ..."
python main_template.py -c "configs/${SUBJECT_NAME}/template.yaml"

echo "Generating position maps ..."
OPENCV_IO_ENABLE_OPENEXR=1 python -m gen_data.gen_pos_maps -c "configs/${SUBJECT_NAME}/avatar.yaml"

echo "Traning avatar ..."
python main_avatar.py -c "configs/${SUBJECT_NAME}/avatar.yaml" --mode=train
