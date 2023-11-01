#!/bin/bash

# Define the number of loops
num_loops=2

# Loop to change the "SCENE" parameter and run the pipeline
for ((i=0; i<num_loops; i++)); do
    # Construct the new "SCENE" value with leading zeros
    scene=$(printf "%02d" $i)

    # Use jq to modify the "SCENE" parameter in config.json
    jq --arg new_scene "$scene" '.SCENE = $new_scene' config.json > temp.json
    mv temp.json config.json

    # Run the pipeline using the modified config.json
    echo "Running synthetic data pipeline for scene: ${scene}"
    ./run.sh
done
