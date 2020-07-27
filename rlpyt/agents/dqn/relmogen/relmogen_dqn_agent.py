
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.models.dqn.relmogen_dqn_model import RelMoGenDqnModel
from rlpyt.agents.dqn.relmogen.mixin import DictObsMixin


class RelMoGenDqnAgent(DictObsMixin, DqnAgent):

    def __init__(self, ModelCls=RelMoGenDqnModel,
                 distribution_type='epsilon', **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
        self.distribution_type = distribution_type

    def initialize(self, env_spaces, share_memory=False,
                   global_B=1, env_ranks=None):
        """Along with standard initialization, creates vector-valued epsilon
        for exploration, if applicable, with a different epsilon for each
        environment instance."""
        super().initialize(env_spaces, share_memory,
                           global_B=global_B, env_ranks=env_ranks)
        self.distribution.embodiment_mode = self.model.embodiment_mode
