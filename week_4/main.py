import argparse
from image_plotter import Plotter
import multiprocessing
parser = argparse.ArgumentParser(description='Create workspace,config space plots.')

parser.add_argument("--data_directory",help="directory to save images")
parser.add_argument("--num_images", help="Number of worksapce, c_obs image pairs to be generated")
parser.add_argument("--num_obstacles", help="number of obstacles in workspace")
parser.add_argument("--obstacle_radii", help="radii of each obstacle")
parser.add_argument("--arm_1_length", help="arm_1_length")
parser.add_argument("--arm_2_length", help="arm_2_length")
parser.add_argument("--arm_width", help="arm_1_width")
parser.add_argument("--height_workspace", help="height_workspace")
parser.add_argument("--width_workspace", help="width_workspace")

if __name__ == '__main__':
    args = parser.parse_args()
    plotter = Plotter(args)
    pool = multiprocessing.Pool(5)
    pool.map(plotter.plot_workspace_cobs_pairs,[(0,200),(201,400),(401,600),(601,800),(801,1000)])