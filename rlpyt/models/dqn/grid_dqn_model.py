import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.deconv2d import Deconv2dModel
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel


class GridDqnModel(torch.nn.Module):
    """Standard convolutional network for DQN.  2-D convolution for multiple
    video frames per observation, feeding an MLP for Q-value outputs for
    the action set.
    """

    def __init__(
            self,
            image_shape,
            output_size,
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
        c, _, _ = image_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels or [16, 32, 32, 64, 64],
            kernel_sizes=kernel_sizes or [3, 3, 3, 3, 3],
            strides=strides or [2, 2, 2, 2, 2],
            paddings=paddings or [1, 1, 1, 1, 1],
        )
        self.bottleneck = Conv2dModel(
            in_channels=64,
            channels=channels or [64, 64],
            kernel_sizes=kernel_sizes or [3, 3],
            strides=strides or [1, 1],
            paddings=paddings or [1, 1],
        )
        self.deconv = Deconv2dModel(
            in_channels=64,
            channels=channels or [64, 64, 32],
            kernel_sizes=kernel_sizes or [3, 3, 3],
            strides=strides or [2, 2, 2],
            paddings=paddings or [1, 1, 1],
            output_paddings=output_paddings or [1, 1, 1],
        )
        self.output = torch.nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1)

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
        img = observation

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
        # print('conv', conv_out.shape)

        bottleneck_out = conv_out
        # bottleneck_out = self.bottleneck(conv_out)
        # print('bottleneck', bottleneck_out.shape)

        deconv_out = bottleneck_out
        # deconv_out = self.deconv(bottleneck_out)
        # print('deconv', deconv_out.shape)

        q = self.output(deconv_out)
        # print('q before flatten', q.shape)

        q = q.view(T * B, -1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)

        # print('q', q.shape)
        # assert False

        return q
