import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.deconv2d import Deconv2dModel
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel
from IPython import embed
import cv2
import numpy as np


class GibsonDqnModel(torch.nn.Module):
    """Standard convolutional network for DQN.  2-D convolution for multiple
    video frames per observation, feeding an MLP for Q-value outputs for
    the action set.
    """

    def __init__(
        self,
        observation_space,
        use_maxpool=False,
        channels=None,  # None uses default.
        kernel_sizes=None,
        strides=None,
        paddings=None,
        output_paddings=None,
        base_only=False,
        draw_path_on_map=False,
        draw_objs_on_map=False,
        feature_fusion=False
    ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.observation_space = observation_space

        # self.sensor_input_dim = 22

        # self.sensor_embedder = MlpModel(
        #     input_size=self.sensor_input_dim,
        #     hidden_sizes=[128, 128]
        # )

        self.base_only = base_only
        self.draw_path_on_map = draw_path_on_map
        self.draw_objs_on_map = draw_objs_on_map
        self.feature_fusion = feature_fusion

        if not self.base_only:
            self.vision_input_dim = 4
            self.vision_deconv_dim = 256 if self.feature_fusion else 128
            self.vision_output_dim = 12
            # self.vision_conv = Conv2dModel(
            #     in_channels=self.vision_input_dim,
            #     channels=channels or [16, 16, 32, 32, 64, 64, 128, 128, 128, 128],
            #     kernel_sizes=kernel_sizes or [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            #     strides=strides or [2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            #     paddings=paddings or [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            # )
            self.vision_conv = Conv2dModel(
                in_channels=self.vision_input_dim,
                channels=channels or [32, 64, 64, 128],
                kernel_sizes=kernel_sizes or [3, 3, 3, 3],
                strides=strides or [2, 2, 2, 2],
                paddings=paddings or [1, 1, 1, 1],
            )
            self.vision_deconv = Deconv2dModel(
                in_channels=self.vision_deconv_dim,
                channels=channels or [128, 64],
                kernel_sizes=kernel_sizes or [3, 3],
                strides=strides or [2, 2],
                paddings=paddings or [1, 1],
                output_paddings=output_paddings or [1, 1],
            )
            self.vision_output = torch.nn.Conv2d(
                in_channels=64,
                out_channels=self.vision_output_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        self.occ_grid_input_dim = 2 if (
            self.draw_path_on_map or self.draw_objs_on_map) else 1
        self.rotate_occ_grid = False
        self.occ_grid_deconv_dim = 256 if self.feature_fusion else 128
        if self.rotate_occ_grid:
            self.occ_grid_output_dim = 1
        else:
            self.occ_grid_output_dim = 12
        self.base_orn_num_bins = 12

        # self.occ_grid_conv = Conv2dModel(
        #     in_channels=self.occ_grid_input_dim,
        #     channels=channels or [16, 16, 32, 32, 64, 64, 128, 128, 128, 128],
        #     kernel_sizes=kernel_sizes or [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        #     strides=strides or [2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
        #     paddings=paddings or [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # )
        self.occ_grid_conv = Conv2dModel(
            in_channels=self.occ_grid_input_dim,
            channels=channels or [32, 64, 64, 128],
            kernel_sizes=kernel_sizes or [3, 3, 3, 3],
            strides=strides or [2, 2, 2, 2],
            paddings=paddings or [1, 1, 1, 1],
        )
        self.occ_grid_deconv = Deconv2dModel(
            in_channels=self.occ_grid_deconv_dim,
            channels=channels or [128, 64],
            kernel_sizes=kernel_sizes or [3, 3],
            strides=strides or [2, 2],
            paddings=paddings or [1, 1],
            output_paddings=output_paddings or [1, 1],
        )
        self.occ_grid_output = torch.nn.Conv2d(
            in_channels=64,
            out_channels=self.occ_grid_output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # self.idx = 0

    def forward(self, observation, prev_action, prev_reward):
        """
        Compute action Q-value estimates from input state.
        Infers leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Convolution layers process as [T*B,
        image_shape[0], image_shape[1],...,image_shape[-1]], with T=1,B=1 when not given.  Expects uint8 images in
        [0,255] and converts them to float32 in [0,1] (to minimize image data
        storage and transfer).  Used in both sampler and in algorithm (both
        via the agent).
        """
        # lead_dim, T, B, input_shape = infer_leading_dims(
        #     observation.sensor, 1)
        # sensor_feature = self.sensor_embedder(
        #     observation.sensor.view(T * B, *input_shape))

        if not self.base_only:
            lead_dim, T, B, input_shape = infer_leading_dims(
                observation.rgbd, 3)
            vision_feature = self.vision_conv(
                observation.rgbd.view(T * B, *input_shape))

        lead_dim, T, B, input_shape = infer_leading_dims(
            observation.occupancy_grid, 3)
        if self.rotate_occ_grid:
            occ_grid_input = observation.occupancy_grid.view(
                T * B * self.base_orn_num_bins,
                *[self.occ_grid_input_dim, input_shape[1], input_shape[2]])
        else:
            occ_grid_input = observation.occupancy_grid.view(
                T * B,
                *input_shape)
        occ_grid_feature = self.occ_grid_conv(occ_grid_input)

        # sensor_feature = sensor_feature.unsqueeze(2).unsqueeze(3).repeat(
        #     1, 1, occ_grid_feature.shape[2], occ_grid_feature.shape[3])

        # Feature fusion
        if self.feature_fusion and (not self.base_only):
            vision_feature = torch.cat(
                [vision_feature, occ_grid_feature], dim=1)
            occ_grid_feature = vision_feature

        if not self.base_only:
            vision_feature = self.vision_deconv(vision_feature)
            vision_q_values = self.vision_output(vision_feature)

        occ_grid_feature = self.occ_grid_deconv(occ_grid_feature)
        occ_grid_q_values = self.occ_grid_output(occ_grid_feature)

        # Q values visualization
        # occ_grid_q_values_np = occ_grid_q_values.cpu().numpy().squeeze(0)
        # occ_grid_q_values_np = \
        #     (occ_grid_q_values_np - np.min(occ_grid_q_values_np)) / \
        #     (np.max(occ_grid_q_values_np) - np.min(occ_grid_q_values_np))
        # for i in range(occ_grid_q_values_np.shape[0]):
        #     pred = occ_grid_q_values_np[i]
        #     cv2.imwrite('vis/button_door/{}_pred_base_{}.png'.format(self.idx, i),
        #                 (pred * 255).astype(np.uint8))

        # vision_q_values_np = vision_q_values.cpu().numpy().squeeze(0)
        # vision_q_values_np = \
        #     (vision_q_values_np - np.min(vision_q_values_np)) / \
        #     (np.max(vision_q_values_np) - np.min(vision_q_values_np))
        # for i in range(vision_q_values_np.shape[0]):
        #     pred = vision_q_values_np[i]
        #     cv2.imwrite('vis/button_door/{}_pred_arm_{}.png'.format(self.idx, i),
        #                 (pred * 255).astype(np.uint8))

        # occ_grid_np = observation.occupancy_grid.cpu().numpy().squeeze(0)
        # grid = np.concatenate((occ_grid_np[0], occ_grid_np[1]), axis=1)
        # cv2.imwrite('vis/button_door/{}_input_occ_grid.png'.format(self.idx),
        #             (grid * 255).astype(np.uint8))

        # rgb_np = observation.rgbd.cpu().numpy()[:3].transpose(1, 2, 0)
        # rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('vis/button_door/{}_input_rgb.png'.format(self.idx),
        #             (rgb_np * 255).astype(np.uint8))

        if not self.base_only:
            vision_q_values = vision_q_values.view(T * B, -1)

        occ_grid_q_values = occ_grid_q_values.view(T * B, -1)

        # first base, then arm
        if not self.base_only:
            q_value = torch.cat([occ_grid_q_values, vision_q_values], axis=-1)
        else:
            q_value = occ_grid_q_values

        q_value = restore_leading_dims(q_value, lead_dim, T, B)

        # self.idx += 1
        return q_value
