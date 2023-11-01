# create some synthetic data
echo "Running the synthetic data pipeline"
blenderproc run src/synthetic_data_pipeline_st.py

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
ANNOTATED_FRAMES="$DATA_DIR/scene_$SCENE-annotate/onepose_data/annotated_frames"
ONEPOSE_DATA="$DATA_DIR/scene_$SCENE-annotate/onepose_data"
BOP_DATA="$DATA_DIR/scene_$SCENE-annotate/bop_data/train_pbr/000000"

# create the animation gif
# convert -delay 2 -loop 0 "$ANNOTATED_FRAMES/*.png" "$ONEPOSE_DATA/synthetic_data_annotated.gif"
ffmpeg -framerate 30 -i $ANNOTATED_FRAMES/%06d.png -c:v libx264 -pix_fmt yuv420p $ONEPOSE_DATA/synthetic_data_annotated.mp4
ffmpeg -framerate 30 -i $BOP_DATA/rgb/%06d.png -c:v libx264 -pix_fmt yuv420p $ONEPOSE_DATA/Frames.m4v

# remove the separate frames again
rm -rf $ANNOTATED_FRAMES

# remove the BOP_DATA dir
rm -rf $DATA_DIR/scene_$SCENE-annotate/bop_data

# remove the depth frames

# move the content of $ONEPOSE_DATA one level up and remove the now empty $ONEPOSE_DATA dir
mv $ONEPOSE_DATA/* $DATA_DIR/scene_$SCENE-annotate/
rmdir $ONEPOSE_DATA

echo "Done"