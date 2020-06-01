import gym
import numpy as np
import os
import yaml
import pybullet as p
import time
import cv2
import argparse
from IPython import embed
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R

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
    get_link_pose, link_from_name, HideOutput, get_pose, wait_for_user, dump_world, plan_nonholonomic_motion, \
    set_point, create_box, stable_z, control_joints, set_base_values_with_z

import torch
from rlpyt.agents.dqn.relmogen.relmogen_dqn_agent import RelMoGenDqnAgent
import collections


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
        use_base=False,
        occ_map_size=256,
        occ_map_crop_range=2.0,
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
        assert self.image_height == self.image_width
        self.use_base = use_base
        self.occ_map_size = occ_map_size
        self.occ_map_crop_range = occ_map_crop_range
        self.downsize_ratio = downsize_ratio
        self.load_scene()
        self.load_observation_action_spaces()
        if self.mode == 'gui':
            cv2.namedWindow('rgb')
            cv2.namedWindow('depth')
            cv2.namedWindow('q')
            cv2.namedWindow('q_before_resize')
            if self.use_base:
                cv2.namedWindow('occ_map')

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
                    jointLowerLimit, jointUpperLimit, _, _, _, _, _, _, _ = p.getJointInfo(obj.body_id, joint_id)
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

        if self.use_base:
            # TODO: increase initial pos range
            self.initial_pos_range = np.array([[0.0, 1.0], [1.0, 1.0]])
        else:
            self.initial_pos_range = np.array([[0.0, 0.0], [1.0, 1.0]])
        self.initial_orn_range = np.array(
            [np.pi - np.pi / 6.0, np.pi + np.pi / 6.0])
        self.debug_line_id = None

    def load_observation_action_spaces(self):
        observation_space = OrderedDict()
        self.vision_space = gym.spaces.Box(low=0.0,
                                           high=1.0,
                                           shape=(4, self.image_height, self.image_width),
                                           dtype=np.float32)
        observation_space['vision'] = self.vision_space
        if self.use_base:
            self.occ_map_space = gym.spaces.Box(low=0.0,
                                                high=1.0,
                                                shape=(1, self.occ_map_size, self.occ_map_size),
                                                dtype=np.float32)
            observation_space['occ_map'] = self.occ_map_space
        self.observation_space = gym.spaces.Dict(observation_space)
        self.image_height_downsized = self.image_height // self.downsize_ratio
        self.image_width_downsized = self.image_width // self.downsize_ratio
        self.occ_map_size_downsized = self.occ_map_size // self.downsize_ratio
        action_range = self.image_height_downsized * self.image_width_downsized
        if self.use_base:
            action_range += self.occ_map_size_downsized * self.occ_map_size_downsized
        self.action_space = gym.spaces.Discrete(action_range)

    def get_occupancy_map(self, scan=None):
        laser_angular_half_range = self.laser_angular_range / 2.0
        angle = np.arange(-laser_angular_half_range / 180 * np.pi,
                          laser_angular_half_range / 180 * np.pi,
                          self.laser_angular_range / 180.0 * np.pi / self.n_horizontal_rays)
        if scan is None:
            hit_fraction = self.get_scan()
        else:
            hit_fraction = scan

        x = self.laser_linear_range * hit_fraction[:, 0] * np.sin(angle)
        y = self.laser_linear_range * hit_fraction[:, 0] * np.cos(angle)
        y = -y

        x = np.concatenate([[0], x, [0]])
        y = np.concatenate([[0], y, [0]])

        img = np.zeros((self.occ_map_size, self.occ_map_size), np.uint8)
        pts = []
        for i in range(len(x)):
            pts.append([(
                int(x[i]+ self.occ_map_crop_range / 2) * self.occ_map_size / self.occ_map_crop_range),
                int((y[i] + self.occ_map_crop_range / 2) * self.occ_map_size / self.occ_map_crop_range
            )])
        pts = np.array(pts, np.int32)
        pts = pts.reshape((1, -1, 1, 2))
        cv2.fillPoly(img, pts, True, 1)
        return img.astype(np.float32)

    def get_state(self):
        state = super().get_state()
        rgb = state['rgb']
        depth = state['depth']
        scan = state['scan']
        rgbd = np.concatenate((rgb, depth), axis=2)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        state = {}
        state['vision'] = rgbd
        if self.use_base:
            state['occ_map'] = np.expand_dims(self.get_occupancy_map(scan), axis=0)
        return state
    
    def occ_rc_to_local_xy(self, occ_rc):
        occ_row, occ_col = occ_rc
        x = -(occ_row / self.occ_map_size - 0.5)  * self.occ_map_crop_range
        y = -(occ_col / self.occ_map_size - 0.5)  * self.occ_map_crop_range
        return [x, y]

    def global_to_local(self, pos, cur_pos, cur_rot):
        return rotate_vector_3d(pos - cur_pos, *cur_rot)

    def local_to_global(self, pos, cur_pos, cur_rot):
        r, p, y = cur_rot
        local_to_global = R.from_euler('xyz', [r, p, y]).as_dcm()
        return np.dot(local_to_global, pos) + cur_pos

    def step_arm(self, action, action_map=None):
        # print('arm')
        row = action // self.image_width_downsized
        col = action % self.image_width_downsized
        image_3d = self.simulator.renderer.render_robot_cameras(modes=('3d'))[0]
        image_row = int((row + 0.5) * self.downsize_ratio)
        image_col = int((col + 0.5) * self.downsize_ratio)
        position_cam = image_3d[image_row, image_col]
        position_world = np.linalg.inv(self.simulator.renderer.V).dot(position_cam)
        position_eye = self.robots[0].eyes.get_position()

        # print('get point:', time.time() - start)
        # start = time.time()

        object_id, link_id, _, hit_pos, hit_normal = p.rayTest(position_eye, position_world[:3] + position_world[:3] - position_eye)[0]
        # print('ray trace:', time.time() - start)
        # start = time.time()

        # maximum push range 1.5m
        valid_range = np.linalg.norm(position_world[:3] - position_eye) < 1.5
        valid_force = valid_range and (object_id, link_id) in self.interactive_objs_joints
        if valid_force:
            pass
            # print('dist', np.linalg.norm(position_world[:3] - position_eye))
            # print('valid arm')

        if self.mode == 'gui':
            if self.debug_line_id is not None:
                self.debug_line_id = p.addUserDebugLine(
                    position_eye, position_world[:3], lineWidth=3, replaceItemUniqueId=self.debug_line_id)
            else:
                self.debug_line_id = p.addUserDebugLine(
                    position_eye, position_world[:3], lineWidth=3)

            rgb = self.simulator.renderer.render_robot_cameras(modes=('rgb'))[0][:, :, :3]
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            depth = -self.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:, :, 2:3]
            depth = np.clip(depth / 5.0, 0.0, 1.0)
            rgb = cv2.circle(rgb, (image_col, image_row),
                                10, (0, 0, 255), 2)
            depth = cv2.circle(depth, (image_col, image_row), 10, (0, 0, 0), 2)
            cv2.imshow('rgb', rgb)
            cv2.imshow('depth', depth)
            if action_map is not None:
                action_map = cv2.circle(
                    action_map, (image_col, image_row), 10, (0, 0, 0), 2)
                action_map = (action_map - np.min(action_map)) / \
                    (np.max(action_map) - np.min(action_map))
                action_map = np.reshape(action_map, (32, 32))
                cv2.imshow('q_before_resize', action_map)
                action_map = cv2.resize(action_map, (256, 256))
                cv2.imshow('q', action_map)
            time.sleep(1.0)

        for _ in range(self.simulator_loop):
            if valid_force:
                p.applyExternalForce(
                    object_id, link_id, -np.array(hit_normal) * 1000, hit_pos, p.WORLD_FRAME)
            self.simulator_step()
        self.simulator.sync()

    def step_base(self, action, action_map=None):
        # print('base')
        action -= self.image_height_downsized * self.image_width_downsized
        row = action // self.occ_map_size_downsized
        col = action % self.occ_map_size_downsized

        occ_map = self.get_occupancy_map()
        if occ_map[row, col] > 0:  # empty space
            # print('valid base')
            # print('rc', row, col)
            image_row = int((row + 0.5) * self.downsize_ratio)
            image_col = int((col + 0.5) * self.downsize_ratio)
            robot_frame_xy = self.occ_rc_to_local_xy([image_row, image_col])
            # print('robot_frame_xy', robot_frame_xy)
            if self.mode == 'gui': 
                occ_map = cv2.circle(occ_map, (image_col, image_row), 10, (0, 0, 255), 2)
                cv2.imshow('occ_map', occ_map)
                time.sleep(5.0)

            #print(image_row, image_col)
            # print(robot_frame_xy)
            cur_rot = self.robots[0].get_rpy()
            cur_pos = self.robots[0].get_position()
            world_frame_base_position = self.local_to_global(np.array([robot_frame_xy[0], robot_frame_xy[1], 0]), cur_pos, cur_rot)
            # print('world_frame_base_position', world_frame_base_position)
            set_base_values_with_z(self.robots[0].robot_ids[0], [world_frame_base_position[0], world_frame_base_position[1], np.pi], z=0)
            self.simulator.sync()

    def step(self, action, action_map=None):
        # print('step:', self.current_step)
        # total_start = time.time()
        # start = time.time()
        assert 0 <= action < self.action_space.n

        self.current_step += 1

        if action < self.image_height_downsized * self.image_width_downsized:
            self.step_arm(action, action_map)
        else:
            self.step_base(action, action_map)

        state = self.get_state()
        # print('get state:', time.time() - start)
        # start = time.time()

        reward = self.get_reward()

        # print('get reward:', time.time() - start)
        # start = time.time()

        done = self.current_step >= self.max_step
        info = {}
        # print('total:', time.time() - total_start)
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
                          downsize_ratio=8,
                          use_base=True,
                          )
    # from IPython import embed
    # embed()

    eval_policy = True
    if eval_policy:
        snapshot_pth = '/cvgl2/u/chengshu/rlpyt/data/local/20200521/222546_copy/relmogen/run_0/itr_26519.pkl'
        # snapshot_pth = '/cvgl2/u/chengshu/rlpyt/data/local/20200529/002054/relmogen/run_0/'
        data = torch.load(snapshot_pth)
        model_state_dict = data['agent_state_dict']['model']
        agent = RelMoGenDqnAgent(initial_model_state_dict=model_state_dict)
        EnvSpaces = collections.namedtuple(
            'EnvSpaces', ['observation', 'action'])
        env_spaces = EnvSpaces(
            observation=nav_env.observation_space, action=nav_env.action_space)
        agent.initialize(env_spaces)
        model = agent.model.eval()

    avg_episode_return = 0.0
    num_episodes = 100
    for episode in range(num_episodes):
        print('Episode: {}'.format(episode))
        start = time.time()
        state = nav_env.reset()
        episode_return = 0.0
        while True:
            if eval_policy:
                if np.random.random() < 0.0:
                    action = nav_env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = model(torch.from_numpy(state), torch.from_numpy(
                            np.array(nav_env.action_space.sample())), 0)
                        action = action.numpy()
                        max_action = np.argmax(action)
                state, reward, done, _ = nav_env.step(max_action, action)
            else:
                action = nav_env.action_space.sample()
                state, reward, done, _ = nav_env.step(action)
            print('step:', nav_env.current_step, 'reward:', reward)
            episode_return += reward
            if done:
                break
        print('Episode finished after {} timesteps, took {} seconds.'.format(
            nav_env.current_step, time.time() - start))
        print('Return: {}'.format(episode_return))
        avg_episode_return += episode_return
    print('Average Return over {} episodes: {}'.format(
        num_episodes, avg_episode_return / num_episodes))
    nav_env.clean()
