import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.deconv2d import Deconv2dModel
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel


class RelMoGenDqnModel(torch.nn.Module):
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
            ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()

        self.modalities = observation_space.names
        self.in_channels = [space.shape[0] for space in observation_space.spaces]
        self.out_channels = [1 for _ in range(len(self.modalities))]
        self.convs = torch.nn.ModuleList([
            Conv2dModel(
                in_channels=in_channel,
                channels=channels or [16, 16, 32, 32, 64, 64],
                kernel_sizes=kernel_sizes or [3, 3, 3, 3, 3, 3],
                strides=strides or [2, 1, 2, 1, 2, 1],
                paddings=paddings or [1, 1, 1, 1, 1, 1],
            ) for in_channel in self.in_channels
        ])
        self.outputs = torch.nn.ModuleList([
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            ) for out_channel in self.out_channels
        ])

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
        q_values = []
        for modality, conv, output in zip(self.modalities, self.convs, self.outputs):
            obs = getattr(observation, modality)
            # Infer (presence of) leading dimensions: [T,B], [B], or [].
            lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)
            conv_out = conv(obs.view(T * B, *img_shape))
            q = output(conv_out)
            q = q.view(T * B, -1)
            # Restore leading dimensions: [T,B], [B], or [], as input.
            q = restore_leading_dims(q, lead_dim, T, B)
            q_values.append(q)

        if len(self.modalities) > 1:
            q_value = torch.cat(q_values, axis=-1)
        else:
            q_value = q_values[0]

        return q_value