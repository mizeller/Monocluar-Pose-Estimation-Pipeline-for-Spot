#!/bin/bash

# this small script, converts some left-over data from the
# old synthetic data pipeline - adapt to your needs if necessary
root_dir="/Users/mizeller/projects/BlenderProc/data_copy"

# Iterate through each scene directory
for scene_dir in $root_dir/*; do
    if [ -d "$scene_dir" ]; then
        for bop_path in "$scene_dir/bop_data/train_pbr"/*; do
            if [ -d "$bop_path" ]; then
                bw_path="$bop_path/bw"
                onepose_path="$scene_dir/onepose_data"
                # Check if bw directory exists and onepose_data directory exists
                if [ -d "$bw_path" ] && [ -d "$onepose_path" ]; then
                    # Convert images to video using ffmpeg
                    ffmpeg -framerate 30 -pattern_type glob -i "$bw_path/*.png" -c:v libx264 "$onepose_path/Frames.m4v"
                    # Remove the bop_data directory
                    rm -r "$bop_path"
                    # move the onepose dir to the root
                    mv "$onepose_path" "$scene_dir-annotate"
                    # and remove the old scene dir
                    rm -r "$scene_dir"
                fi
            fi
        done
    fi
done