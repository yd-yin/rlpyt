
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.models.dqn.relmogen_dqn_model import RelMoGenDqnModel
from rlpyt.agents.dqn.relmogen.mixin import DictObsMixin


class RelMoGenDqnAgent(DictObsMixin, DqnAgent):

    def __init__(self, ModelCls=RelMoGenDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
