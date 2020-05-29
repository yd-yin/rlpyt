import gym
import numpy as np
import os
import yaml
import pybullet as p
import time
import cv2
import argparse
from IPython import embed

import gibson2
from gibson2.core.physics.robot_locomotors import Turtlebot, JR2_Kinova, Fetch
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import EmptyScene
from gibson2.core.physics.interactive_objects import InteractiveObj, BoxShape, YCBObject, VisualMarker
from gibson2.utils.utils import parse_config
from gibson2.core.render.profiler import Profiler
from gibson2.envs.locomotor_env import NavigateEnv
from gibson2.utils.utils import parse_config, rotate_vector_3d, l2_distance, quatToXYZW, cartesian_to_polar
from transforms3d.euler import euler2quat

from gibson2.external.pybullet_tools.utils import set_base_values, joint_from_name, set_joint_position, \
    set_joint_positions, add_data_path, connect, plan_base_motion, plan_joint_motion, enable_gravity, \
    joint_controller, dump_body, load_model, joints_from_names, user_input, disconnect, get_joint_positions, \
    get_link_pose, link_from_name, HideOutput, get_pose, wait_for_user, dump_world, plan_nonholonomic_motion, set_point, create_box, stable_z, control_joints


class RelMoGenEnv(NavigateEnv):
    def __init__(
        self,
        config_file,
        model_id=None,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        automatic_reset=False,
        device_idx=0,
        render_to_tensor=False,
        downsize_ratio=8,
    ):
        """
        :param config_file: config_file path
        :param model_id: override model_id in config file
        :param mode: headless or gui mode
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param automatic_reset: whether to automatic reset after an episode finishes
        :param random_height: whether to randomize height for target position (for reaching task)
        :param device_idx: device_idx: which GPU to run the simulation and rendering on
        """
        super().__init__(config_file,
                         model_id=model_id,
                         mode=mode,
                         action_timestep=action_timestep,
                         physics_timestep=physics_timestep,
                         automatic_reset=automatic_reset,
                         device_idx=device_idx,
                         render_to_tensor=render_to_tensor)
        self.downsize_ratio = downsize_ratio
        self.load_scene()
        self.load_observation_action_spaces()
        # cv2.namedWindow('rgb')
        # cv2.namedWindow('depth')

    def load_scene(self):
        interactive_objs = []
        interactive_objs_joints = []
        interactive_objs_joint_limits = []
        interactive_objs_positions = []

        interactive_objs.append(InteractiveObj(filename=os.path.join(
            gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf')))
        interactive_objs_positions.append([-2.0, 0.4, 0.5])

        interactive_objs.append(InteractiveObj(filename=os.path.join(
            gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf')))
        interactive_objs_positions.append([-2.0, 1.6, 0.5])

        interactive_objs.append(InteractiveObj(filename=os.path.join(
            gibson2.assets_path, 'models/cabinet/cabinet_0004.urdf')))
        interactive_objs_positions.append([-2.1, 1.6, 2.0])

        interactive_objs.append(InteractiveObj(filename=os.path.join(
            gibson2.assets_path, 'models/cabinet/cabinet_0004.urdf')))
        interactive_objs_positions.append([-2.1, 0.4, 2.0])

        for obj, pos in zip(interactive_objs, interactive_objs_positions):
            self.simulator.import_articulated_object(obj)
            obj.set_position(pos)
            for joint_id in range(p.getNumJoints(obj.body_id)):
                _, _, jointType, _, _, _, _, _, \
                    jointLowerLimit, jointUpperLimit, _, _, _, _, _, _, _ = p.getJointInfo(
                        obj.body_id, joint_id)
                if jointType == p.JOINT_REVOLUTE or jointType == p.JOINT_PRISMATIC:
                    interactive_objs_joints.append((obj.body_id, joint_id))
                    interactive_objs_joint_limits.append(
                        (obj.body_id, joint_id, jointLowerLimit, jointUpperLimit))

        obj = BoxShape(pos=[-2.05, 1.0, 0.5], dim=[0.35, 0.2, 0.5])
        self.simulator.import_object(obj)
        obj = BoxShape(pos=[-2.45, 1.0, 1.5], dim=[0.01, 2.0, 1.5])
        self.simulator.import_object(obj)
        p.createConstraint(0, -1, obj.body_id, -1, p.JOINT_FIXED,
                           [0, 0, 1], [-2.55, 1.0, 1.5], [0.0, 0.0, 0.0])

        self.interactive_objs = interactive_objs
        self.interactive_objs_joints = interactive_objs_joints
        self.interactive_objs_joint_limits = interactive_objs_joint_limits

        self.initial_pos_range = np.array([[0.0, 0.0], [1.0, 1.0]])
        self.initial_orn_range = np.array(
            [np.pi - np.pi / 6.0, np.pi + np.pi / 6.0])
        self.hit = 0
        self.debug_line_id = None

    def load_observation_action_spaces(self):
        self.observation_space = gym.spaces.Box(low=0.0,
                                                high=1.0,
                                                shape=(
                                                    4, self.image_height, self.image_height),
                                                dtype=np.float32)
        self.image_height_downsized = self.image_height // self.downsize_ratio
        self.image_width_downsized = self.image_width // self.downsize_ratio
        self.action_space = gym.spaces.Discrete(
            self.image_height_downsized * self.image_width_downsized * 2)

    def get_occupancy_map(self):
        laser_angular_half_range = self.laser_angular_range / 2.0
        angle = np.arange(-laser_angular_half_range / 180 * np.pi,
                                            laser_angular_half_range / 180 * np.pi,
                                            self.laser_angular_range / 180.0 * np.pi / self.n_horizontal_rays)
        hit_fraction = self.get_scan()
        x = self.laser_linear_range * hit_fraction[:,0] * np.sin(angle)
        y = self.laser_linear_range * hit_fraction[:,0] * np.cos(angle)
        y = -y

        x = np.concatenate([[0],x,[0]])
        y = np.concatenate([[0],y,[0]])

        self.occ_map_size = 256
        img = np.zeros((self.occ_map_size, self.occ_map_size), np.uint8)
        self.occ_map_crop_range = 8
        pts = []
        for i in range(len(x)):
            pts.append([(int(x[i]+self.occ_map_crop_range/2) * self.occ_map_size / self.occ_map_crop_range) ,
                            int((y[i] + self.occ_map_crop_range/2) * self.occ_map_size / self.occ_map_crop_range)])
        pts = np.array(pts, np.int32)
        pts = pts.reshape((1,-1,1,2))
        cv2.fillPoly(img,pts,True,1)

        return img.astype(np.float32)


    def get_state(self):
        rgb = self.get_rgb()
        depth = self.get_depth()
        rgbd = np.concatenate((rgb, depth), axis=2)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        state = {}
        state['rgbd'] = rgbd
        state['occupancy_map'] = self.get_occupancy_map()
        # rgb = self.simulator.renderer.render_robot_cameras(modes=('rgb'))[0][:, :, :3]
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # depth = -self.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:, :, 2:3]
        # depth = np.clip(depth / 5.0, 0.0, 1.0)
        # cv2.imshow('rgb', rgb)
        # cv2.imshow('depth', depth)
        return state

    def occ_rc_to_world_xy(self, occ_rc):
        occ_row, occ_col = occ_rc
        x = -(occ_row / self.occ_map_size - 0.5)  * self.occ_map_crop_range
        y = -(occ_col / self.occ_map_size - 0.5)  * self.occ_map_crop_range
        return [x,y]

    def global_to_local(self, pos, cur_pos, cur_rot):
        return rotate_vector_3d(pos - cur_pos, *cur_rot)
    
    def local_to_global(self, pos, cur_pos, cur_rot):
        return rotate_vector_3d(pos, -cur_rot[0], -cur_rot[1], -cur_rot[2]) + cur_pos
    
    def step(self, action):
        print('step:', self.current_step)
        total_start = time.time()
        # start = time.time()
        assert 0 <= action < self.image_height_downsized * self.image_width_downsized * 2

        self.current_step += 1

        if action < self.image_height_downsized * self.image_width_downsized: # base action
            row = action // self.image_width_downsized
            col = action % self.image_width_downsized
            image_row = int((row + 0.5) * self.downsize_ratio)
            image_col = int((col + 0.5) * self.downsize_ratio)
            robot_frame_xy = self.occ_rc_to_world_xy([image_row, image_col])
            print(image_row, image_col)
            print(robot_frame_xy)
            # reset base position

        else: # arm action

            row = (action - self.image_height_downsized * self.image_width_downsized) // self.image_width_downsized
            col = (action - self.image_height_downsized * self.image_width_downsized) % self.image_width_downsized

            image_3d = self.simulator.renderer.render_robot_cameras(modes=('3d'))[0]
            image_row = int((row + 0.5) * self.downsize_ratio)
            image_col = int((col + 0.5) * self.downsize_ratio)
            position_cam = image_3d[image_row, image_col]
            position_world = np.linalg.inv(
                self.simulator.renderer.V).dot(position_cam)
            position_eye = self.robots[0].eyes.get_position()

            # print('get point:', time.time() - start)
            # start = time.time()

            object_id, link_id, _, hit_pos, hit_normal = p.rayTest(
                position_eye, position_world[:3] + position_world[:3] - position_eye)[0]
            # print('ray trace:', time.time() - start)
            # start = time.time()

            if self.mode == 'gui':
                if self.debug_line_id is not None:
                    self.debug_line_id = p.addUserDebugLine(
                        position_eye, position_world[:3], lineWidth=3, replaceItemUniqueId=self.debug_line_id)
                else:
                    self.debug_line_id = p.addUserDebugLine(
                        position_eye, position_world[:3], lineWidth=3)

            valid_force = (object_id, link_id) in self.interactive_objs_joints

            # if valid_force:
            #     self.hit += 1
            #     print('VALID', (object_id, link_id))
            #     print('BEFORE', self.get_potential())

            for _ in range(self.simulator_loop):
                if valid_force:
                    p.applyExternalForce(
                        object_id, link_id, -np.array(hit_normal) * 1000, hit_pos, p.WORLD_FRAME)
                self.simulator_step()
            self.simulator.sync()
            # print('apply force:', valid_force, time.time() - start)
            # start = time.time()

            # if valid_force:
            #     print('AFTER', self.get_potential())




        state = self.get_state()

        # print('get state:', time.time() - start)
        # start = time.time()

        reward = self.get_reward()

        # print('get reward:', time.time() - start)
        # start = time.time()

        done = self.current_step >= self.max_step
        info = {}
        print('total:', time.time() - total_start)
        return state, reward, done, info

    def reset_agent(self):
        self.initial_pos = np.array([
            np.random.uniform(
                self.initial_pos_range[0][0], self.initial_pos_range[0][1]),
            np.random.uniform(
                self.initial_pos_range[1][0], self.initial_pos_range[1][1]),
            0.0
        ])
        self.initial_orn = np.random.uniform(
            self.initial_orn_range[0], self.initial_orn_range[1])
        self.robots[0].set_position_orientation(pos=self.initial_pos,
                                                orn=quatToXYZW(euler2quat(0, 0, self.initial_orn), 'wxyz'))
        self.robots[0].robot_specific_reset()
        self.robots[0].keep_still()

    def reset_interactive_objects(self):
        for body_id, joint_id, lower_limit, upper_limit in self.interactive_objs_joint_limits:
            joint_pos = np.random.uniform(lower_limit, upper_limit)
            p.resetJointState(body_id, joint_id,
                              targetValue=joint_pos, targetVelocity=0.0)

    def get_reward(self):
        new_potential = self.get_potential()
        potential_reward = self.potential - new_potential
        reward = potential_reward * self.potential_reward_weight
        self.potential = new_potential
        return reward

    def get_potential(self):
        potential = 0.0
        for body_id, joint_id in self.interactive_objs_joints:
            potential += p.getJointState(body_id, joint_id)[0]
        return potential

    def reset(self):
        self.reset_interactive_objects()
        state = super().reset()
        self.potential = self.get_potential()
        self.simulator_step()
        self.simulator.sync()
        return state

    def close(self):
        self.clean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        required=True,
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')
    args = parser.parse_args()

    nav_env = RelMoGenEnv(config_file=args.config,
                          mode=args.mode,
                          action_timestep=1.0 / 24.0,
                          physics_timestep=1.0 / 240.0,
                          device_idx=0,
                          downsize_ratio=8)

    step_time_list = []
    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        nav_env.reset()
        while True:
            action = nav_env.action_space.sample()
            state, reward, done, _ = nav_env.step(action)
            print('step:', nav_env.current_step, 'reward:', reward)
            if done:
                break
        print('Episode finished after {} timesteps, took {} seconds.'.format(
            nav_env.current_step, time.time() - start))
    print(nav_env.hit / (100 * nav_env.max_step))
    nav_env.clean()
