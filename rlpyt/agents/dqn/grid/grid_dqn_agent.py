
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.models.dqn.grid_dqn_model import GridDqnModel
from rlpyt.agents.dqn.atari.mixin import AtariMixin


class GridDqnAgent(AtariMixin, DqnAgent):

    def __init__(self, ModelCls=GridDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
