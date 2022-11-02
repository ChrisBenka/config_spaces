import math
from shapely.geometry import Polygon, Point, LineString, MultiPoint
from shapely import affinity
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils import round_point, round_polygon, round_line_string, midpoint
import random

plt.ioff()


class Plotter:
    def __init__(self, args):
        self.num_images = int(args.num_images)
        self.num_obstacles = int(args.num_obstacles)
        self.obstacle_radii = float(args.obstacle_radii)
        self.arm_1_length = float(args.arm_1_length)
        self.arm_2_length = float(args.arm_2_length)
        self.arm_width = float(args.arm_width)
        self.height_workspace = int(args.height_workspace)
        self.width_workspace = int(args.width_workspace)

        self.origin = Point(0, 0)
        self.total_arm_length = self.arm_1_length + self.arm_2_length

        self.robot_arm1_og = MultiPoint(list([self.origin, Point(self.arm_1_length, 0)
                                                 , Point(0, self.arm_width),
                                              Point(self.arm_1_length, self.arm_width)])).convex_hull
        self.total_arm_og = MultiPoint(list(
            [self.origin, Point(self.total_arm_length, 0), Point(0, self.arm_width),
             Point(self.total_arm_length, .5)])).convex_hull

        self.arm_1_joint = self.origin
        self.workspace_dir = args.data_dir + "workspace/"
        self.configspace_dir = args.data_dir + "cobs/"
        self.include_self_collision = args.include_self_collision
        self.include_axis = args.include_axis

        self.q1 = [0, 360]
        self.q2 = [0, -359]

    def plot_workspace(self, obstacles, plot_id, include_robot=False, q1=0, q2=0, include_labels=False,
                       title="Workspace", file_nm=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        if include_robot:
            arm_1_rot = q1
            robot_arm1 = round_polygon(affinity.rotate(self.robot_arm1_og, arm_1_rot, self.arm_1_joint))
            total_a = round_polygon(affinity.rotate(self.total_arm_og, arm_1_rot, self.arm_1_joint))

            total_a_coords = {*total_a.exterior.coords}
            arm_1_coords = {*robot_arm1.exterior.coords}
            upper_arm_coords = total_a_coords.difference(arm_1_coords).union(arm_1_coords.difference(total_a_coords))
            upper_arm = MultiPoint(list(upper_arm_coords)).convex_hull

            possible_joints = list(arm_1_coords.intersection(upper_arm_coords))
            mid = midpoint(possible_joints[0], possible_joints[1])

            upper_arm_joint = mid

            upper_arm_rotate = affinity.rotate(upper_arm, -q2, upper_arm_joint)

            ax.fill(*robot_arm1.exterior.xy, color='green')
            ax.plot(self.origin.x, self.origin.y, marker="o", markerfacecolor="black", label='q1 - [0,360]')
            ax.plot(upper_arm_joint[0], upper_arm_joint[1], marker="o", markerfacecolor="black", label='q2 - [0,360]')
            ax.fill(*upper_arm_rotate.exterior.xy, color='blue')

        self.plot_polys(obstacles)
        if include_labels:
            ax.legend(bbox_to_anchor=(1, 1))
            ax.set_title(title)
        # if not self.include_axis:
        #     ax.axis("off")
        #     ax.xaxis.set_visible(False)
        #     ax.yaxis.set_visible(False)
        plt.grid()
        if file_nm:
            plt.savefig(f"{self.workspace_dir}{file_nm}",bbox_inches='tight')
        else:
            plt.savefig(f"{self.workspace_dir}{plot_id}",bbox_inches='tight')

        plt.close(fig)  # close the figure window

    def calculate_cobs(self, obstacles):
        c_pts = []
        for i in range(self.q1[0], self.q1[1] + 1):
            c_arm_1_rotated = round_polygon(affinity.rotate(self.robot_arm1_og, i, self.origin))
            c_total_arm_rotated = round_polygon(affinity.rotate(self.total_arm_og, i, self.origin))

            c_total_arm_coords = {*c_total_arm_rotated.exterior.coords}
            c_arm_1_rotated_coords = {*c_arm_1_rotated.exterior.coords}
            c_upper_arm_coords = c_total_arm_coords.difference(c_arm_1_rotated_coords).union(
                c_arm_1_rotated_coords.difference(c_total_arm_coords))
            c_upper_arm = MultiPoint(list(c_upper_arm_coords)).convex_hull

            c_possible_joints = list(c_arm_1_rotated_coords.intersection(c_upper_arm_coords))

            assert len(c_possible_joints) == 2, f"{i},{j}"

            c_upper_arm_joint = midpoint(c_possible_joints[0], c_possible_joints[1])

            for j in range(self.q2[0], self.q2[1] - 1, -1):
                if self.include_self_collision and 170 <= abs(j) <= 190:
                    c_pts.append((i, abs(j), 3))
                    continue
                c_arm_2_rotated = round_polygon(affinity.rotate(c_upper_arm, j, c_upper_arm_joint))
                for obs_id, o in enumerate(obstacles):
                    if c_arm_2_rotated.intersects(o) or c_arm_1_rotated.intersects(o):
                        c_pts.append((i, abs(j), obs_id))
                        break
        return c_pts

    def plot_cobs(self, c_obs, id, group_by_obs=True, include_labels=False, file_nm=None):
        def group_c_obs_by_obs_id(c_obs):
            res = [[], [], [], []]
            for c in c_obs:
                if c[-1] == 0:
                    res[0].append(c[:2])
                elif c[-1] == 1:
                    res[1].append(c[:2])
                elif c[-1] == 2:
                    res[2].append(c[:2])
                else:
                    res[3].append(c[:2])
            return res

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim([0, 360])
        ax.set_ylim([0, 360])

        if include_labels:
            ax.set_title("Config Space")
        if group_by_obs:
            grouped_cobs = group_c_obs_by_obs_id(c_obs)
            for group_id, group in enumerate(grouped_cobs):
                x_s = list(map(lambda pt: pt[0], group))
                y_s = list(map(lambda pt: pt[1], group))
                label = "c-obs-self-collision" if group_id == 3 else f"c_obs-obst{group_id + 1}"
                ax.scatter(x_s, y_s, label=label)
        else:
            x_s = list(map(lambda pt: pt[0], c_obs))
            y_s = list(map(lambda pt: pt[1], c_obs))
            ax.scatter(x_s, y_s, label='c_obs')

        ax.set_xlabel('Q1')
        ax.set_ylabel('Q2')

        if include_labels:
            ax.legend(bbox_to_anchor=(1, 1))
        # if not self.include_axis:
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        if file_nm:
            plt.savefig(f"{self.configspace_dir}{file_nm}", bbox_inches='tight')
        else:
            plt.savefig(f"{self.configspace_dir}{id}", bbox_inches='tight')

        plt.close()

    def plot_workspace_cobs_pairs(self, image_config):
        start_image, end_image = image_config
        if start_image == 0:
            for i in range(1):
                obstacles = self.generate_obstacle_origins()
                self.plot_workspace(obstacles, i, True, include_labels=True, file_nm="workspace_base")
                cobs = self.calculate_cobs(obstacles)
                self.plot_cobs(cobs, i, True, True, file_nm="configspace_base")
        for i in range(start_image, end_image + 1):
            obstacles = self.generate_obstacle_origins()
            self.plot_workspace(obstacles, i, False, include_labels=False)
            cobs = self.calculate_cobs(obstacles)
            self.plot_cobs(cobs, i, True, False)

    def generate_obstacle_origins(self):
        obstacle_origins = set([])
        while len(obstacle_origins) < self.num_obstacles:
            new_obstacle = tuple([round(random.uniform(-3.5, 3.5), 4) for i in range(2)])
            can_add = True
            for obst in obstacle_origins:
                if Point(obst).buffer(self.obstacle_radii).intersects(Point(new_obstacle).buffer(self.obstacle_radii)):
                    can_add = False
                elif Point(self.origin).buffer(self.obstacle_radii).intersects(
                        Point(new_obstacle).buffer(self.obstacle_radii)):
                    can_add = False
            if can_add:
                obstacle_origins.add(new_obstacle)
        obstacles = [Point(pt[0], pt[1]).buffer(self.obstacle_radii) for pt in obstacle_origins]
        return obstacles

    def plot_polys(self, polys):
        for poly_id, poly in enumerate(polys):
            self.plot_coords(poly.exterior.coords)
            plt.fill_between(*poly.exterior.xy, alpha=.5, label=f"obstacle {poly_id + 1}")

    def plot_coords(self, coords):
        pts = list(coords)
        x, y = zip(*pts)
        plt.plot(x, y)
