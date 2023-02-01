import argparse
from project.scripts.image_plotter import Plotter
import multiprocessing
parser = argparse.ArgumentParser(description='Create workspace,config space plots.')

parser.add_argument("--data_dir",help="directory to save images",default="/home/chris/Documents/columbia/fall_22/config_space/config_spaces/project/scripts/data/")
parser.add_argument("--num_images", help="Number of worksapce, c_obs image pairs to be generated",default=10000)
parser.add_argument("--num_obstacles", help="number of obstacles in workspace",default=3)
parser.add_argument("--obstacle_radii", help="radii of each obstacle",default=.5)
parser.add_argument("--arm_1_length", help="arm_1_length",default=2)
parser.add_argument("--arm_2_length", help="arm_2_length",default=2)
parser.add_argument("--arm_width", help="arm_1_width",default=.5)
parser.add_argument("--height_workspace", help="height_workspace",default=10)
parser.add_argument("--width_workspace", help="width_workspace",default=10)
parser.add_argument("--include_self_collision", help="include self collision in cobs",default=True)
parser.add_argument("--include_axis",help="include axis on workspace and cobs images",default=False)

if __name__ == '__main__':
    args = parser.parse_args()
    plotter = Plotter(args)
    pool = multiprocessing.Pool(5)
    pairs = [(1001,2800),(2801,4600),(4601,6400),(6401,8200),(8201,10000)]
    pool.map(plotter.plot_workspace_cobs_pairs,pairs)