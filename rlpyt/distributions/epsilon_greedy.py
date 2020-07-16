
import torch

from rlpyt.distributions.base import Distribution
from rlpyt.distributions.discrete import DiscreteMixin


class EpsilonGreedy(DiscreteMixin, Distribution):
    """For epsilon-greedy exploration from state-action Q-values."""

    def __init__(self, epsilon=1, **kwargs):
        super().__init__(**kwargs)
        self._epsilon = epsilon

    def sample(self, q, observation=None):
        """Input can be shaped [T,B,Q] or [B,Q], and vector epsilon of length
        B will apply across the Batch dimension (same epsilon for all T)."""
        if observation is None or self.embodiment_mode == 'arm':
            arg_select = torch.argmax(q, dim=-1)
            mask = torch.rand(arg_select.shape) < self._epsilon
            arg_rand = torch.randint(
                low=0, high=q.shape[-1], size=(mask.sum(),))
            arg_select[mask] = arg_rand
        else:
            assert len(q.shape) <= 3
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
                    num_valid_idx = valid_idx.shape[0]
                    if torch.rand(1)[0] < self._epsilon:
                        idx = torch.randint(low=0,
                                            high=num_valid_idx,
                                            size=(1,))[0]
                    else:
                        single_q_base = q[t][b][:(12 * 32 * 32)]
                        single_q_base_valid = \
                            single_q_base[single_occ_grid.bool()]

                        if self.embodiment_mode == 'base':
                            single_q_valid = single_q_base_valid
                        else:
                            single_q_arm = q[t][b][(12 * 32 * 32):]
                            single_q_valid = torch.cat(
                                (single_q_base_valid, single_q_arm))
                        idx = torch.argmax(single_q_valid)
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
    def epsilon(self):
        return self._epsilon

    def set_epsilon(self, epsilon):
        """Assign value for epsilon (can be vector)."""
        self._epsilon = epsilon


class CategoricalEpsilonGreedy(EpsilonGreedy):
    """For epsilon-greedy exploration from distributional (categorical)
    representation of state-action Q-values."""

    def __init__(self, z=None, **kwargs):
        super().__init__(**kwargs)
        self.z = z

    def sample(self, p, z=None):
        """Input p to be shaped [T,B,A,P] or [B,A,P], A: number of actions, P:
        number of atoms.  Optional input z is domain of atom-values, shaped
        [P].  Vector epsilon of lenght B will apply across Batch dimension."""
        q = torch.tensordot(p, z or self.z, dims=1)
        return super().sample(q)

    def set_z(self, z):
        """Assign vector of bin locations, distributional domain."""
        self.z = z
