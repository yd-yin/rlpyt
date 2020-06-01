class DictObsMixin:
    def make_env_to_model_kwargs(self, env_spaces):
        return dict(observation_space=env_spaces.observation.space)
