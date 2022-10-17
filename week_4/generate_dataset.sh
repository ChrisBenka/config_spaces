
foldername=$(date +%Y_%m_%d_%s)
mkdir -p  ./data/"$foldername/workspace"
mkdir -p  ./data/"$foldername/cobs"

python image_plotter.py --num_images 10 --num_obstacles 3 --obstacle_radii .5 --arm_1_length 2 --arm_2_length 2 --arm_1_width .5 --arm_2_width .5 --height_workspace 10 --width_workspace 10