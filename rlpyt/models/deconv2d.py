
import torch

from rlpyt.models.mlp import MlpModel
from rlpyt.models.utils import conv2d_output_shape


class Deconv2dModel(torch.nn.Module):
    """2-D Deconvolutional model component.
    Requires number of input channels, but
    not input shape.  Uses ``torch.nn.ConvTranspose2d``.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            output_paddings=None,
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        if output_paddings is None:
            output_paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        deconv_layers = [torch.nn.ConvTranspose2d(
            in_channels=ic,
            out_channels=oc,
            kernel_size=k,
            stride=s,
            padding=p,
            output_padding=op)
            for (ic, oc, k, s, p, op) 
            in zip(in_channels, channels, kernel_sizes, strides, paddings, output_paddings)]
        sequence = list()
        for conv_layer in deconv_layers:
            sequence.extend([conv_layer, nonlinearity()])
        self.deconv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        return self.deconv(input)
