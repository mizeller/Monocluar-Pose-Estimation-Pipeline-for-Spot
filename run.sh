# create some synthetic data
echo "Running the synthetic data pipeline"
blenderproc run src/synthetic_data_pipeline.py

# convert the data from BOP to onePose format
echo "Running the BOP to onePose data conversion"
python src/transform_data.py

# visualize the data
echo "Visualizing the data"
python src/visualise_data.py

# read params from config.json (`brew install jq` if necessary)
echo "Creating the animation gif and removing the separate frames"
DATA_DIR=$(jq -r '.DATA_DIR' config.json)
SCENE=$(jq -r '.SCENE' config.json)
ANNOTATED_FRAMES="$DATA_DIR/scene_$SCENE/onepose_data/annotated_frames"
ONEPOSE_DATA="$DATA_DIR/scene_$SCENE/onepose_data/"

# create the animation gif

convert -delay 2 -loop 0 "$ANNOTATED_FRAMES/*.png" "$ONEPOSE_DATA/synthetic_data_annotated.gif"

# remove the separate frames again
rm -rf $ANNOTATED_FRAMES
echo "Done"