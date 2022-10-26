#!/bin/bash

echo Preprocessing and processing directory of bags

# Pre-preocessing variables
DATASET_DIR=/home/mateo/Data/SARA/arl_data_collect/
TXT_OUTPUT_FILE=/home/mateo/Data/SARA/arl_data_collect/preprocess.txt

# Processing variables
CONFIG_SPEC=/home/mateo/rosbag_to_dataset/specs/warthog5.yaml
SAVE_TO=/home/mateo/Data/SARA/arl_data_collect_dataset/

# Define python version and directories for script
EXE_PYTHON=python3
BASE_DIR=/home/mateo/rosbag_to_dataset/scripts
PREPROCESSING_SCRIPT=preprocess_dataset_2.py
PROCESSING_SCRIPT=convert_bags.py

# Run labeling script
${EXE_PYTHON} $BASE_DIR/$PREPROCESSING_SCRIPT \
    --dataset_dir $DATASET_DIR \
    --output_file $TXT_OUTPUT_FILE

# Run bag processing script
${EXE_PYTHON} $BASE_DIR/$PROCESSING_SCRIPT \
    --config_spec $CONFIG_SPEC \
    --bag_list $TXT_OUTPUT_FILE \
    --save_to $SAVE_TO


echo Pre-processing and processing script ends