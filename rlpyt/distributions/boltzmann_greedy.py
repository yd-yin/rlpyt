
import torch

from rlpyt.distributions.base import Distribution
from rlpyt.distributions.discrete import DiscreteMixin


class BoltzmannGreedy(DiscreteMixin, Distribution):
    """For Boltzmann-greedy exploration from state-action Q-values."""

    def __init__(self, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self._temperature = temperature

    def sample(self, q, observation=None):
        """Input can be shaped [T,B,Q] or [B,Q]."""
        q_shape = len(q.shape)
        if len(q.shape) == 1:
            q = q.unsqueeze(0).unsqueeze(1)
        elif len(q.shape) == 2:
            q = q.unsqueeze(0)

        T, B = q.shape[:2]

        occ_grid = observation.occupancy_grid
        assert len(occ_grid.shape) <= 5
        if len(occ_grid.shape) == 3:
            occ_grid = occ_grid.unsqueeze(0).unsqueeze(1)
        elif len(occ_grid.shape) == 4:
            occ_grid = occ_grid.unsqueeze(0)

        arg_select = []
        for t in range(q.shape[0]):
            for b in range(q.shape[1]):
                if observation is None or self.embodiment_mode == 'arm':
                    action_prob = torch.softmax(
                        q[t][b] / self._temperature, dim=-1)
                    arg_select.append(torch.multinomial(action_prob, 1).item())
                else:
                    single_occ_grid = occ_grid[t][b][0][2::4, 2::4]
                    # single_occ_grid = occ_grid[t][b][0][4::8, 4::8]
                    single_occ_grid = single_occ_grid.flatten().repeat(12)
                    # valid_idx = single_occ_grid.nonzero().squeeze(1)
                    if self.embodiment_mode == 'base':
                        valid_idx = single_occ_grid.nonzero().squeeze(1)
                    else:
                        valid_idx = torch.cat(
                            (single_occ_grid.nonzero().squeeze(1),
                                torch.arange(12 * 32 * 32, 24 * 32 * 32)))
                    single_q_base = q[t][b][:(12 * 32 * 32)]
                    single_q_base_valid = \
                        single_q_base[single_occ_grid.bool()]

                    if self.embodiment_mode == 'base':
                        single_q_valid = single_q_base_valid
                    else:
                        single_q_arm = q[t][b][(12 * 32 * 32):]
                        single_q_valid = torch.cat(
                            (single_q_base_valid, single_q_arm))
                    action_prob = torch.softmax(
                        single_q_valid / self._temperature, dim=-1)
                    idx = torch.multinomial(action_prob, 1).item()
                    original_idx = valid_idx[idx]
                    arg_select.append(original_idx)

        arg_select = torch.stack(arg_select)
        if q_shape == 1:
            arg_select = arg_select[0]
        elif q_shape == 2:
            pass
        elif q_shape == 3:
            arg_select = arg_select.reshape(T, B)

        return arg_select

    @property
    def temperature(self):
        return self._temperature

    def set_epsilon(self, temperature):
        """Assign value for temperature"""
        self._temperature = temperature
