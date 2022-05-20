#!/bin/bash

# TODO updat the below to reflect the locations on your local system
export CITYSCAPES_DATA_ROOT='/seg_data/data/cityscapes'  # Location of your Cityscapes download. See CityscapesSetup.md
export SAVE_DIR='/data/'  # CHANGE TO YOUR PREFERRED SAVE LOCATION
export PROJECTS_ROOT=$HOME  # This dir must contain (not necessarily directly) all of the SUTs. See study_data/README.md
export WORKING_DIR=${SAVE_DIR}'semImFuzz/'


export PYTHONPATH=$PYTHONPATH:$(pwd)