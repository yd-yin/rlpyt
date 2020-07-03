
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""

import os
import torch
import gibson2
import rlpyt
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.envs.gym_wrapper import make_gibson as gym_make
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.grid.grid_dqn_agent import GridDqnAgent
from rlpyt.agents.dqn.relmogen.relmogen_dqn_agent import RelMoGenDqnAgent
from rlpyt.models.dqn.gibson_dqn_model import GibsonDqnModel
from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.replays.non_sequence.uniform import UniformReplayBuffer


def build_and_train(run_ID=0,
                    arena='push_door',
                    cuda_idx=None,
                    model_path=None,
                    eval_only=False):

    gibson_cfg = os.path.join(
        os.path.dirname(gibson2.__file__),
        '../examples/configs/fetch_interactive_nav_s2r_mp.yaml')
    print(gibson_cfg)

    env_cfg = dict(
        mode='headless',
        config_file=gibson_cfg,
        action_timestep=1 / 500.0,
        physics_timestep=1 / 500.0,
        arena=arena,
        action_map=True,
        channel_first=True,
        draw_path_on_map=True,
        base_only=False,
        rotate_occ_grid=False,
        device_idx=0,
    )

    if eval_only:
        n_steps = 0
        snapshot_mode = 'none'
        num_train_env = 0
        num_eval_env = 1
    else:
        n_steps = int(1e6)
        snapshot_mode = 'all'
        num_train_env = 10
        num_eval_env = 2

    num_cpus = num_train_env + num_eval_env
    affinity = dict(
        cuda_idx=cuda_idx,
        workers_cpus=list(os.sched_getaffinity(0))[:num_cpus],
        set_affinity=False
    )

    if model_path is not None:
        assert os.path.isfile(model_path), \
            'model_path does not exist: {}'.format(model_path)
        data = torch.load(model_path)
        agent_state_dict = data['agent_state_dict']['model']
        optimizer_state_dict = data['optimizer_state_dict']
    else:
        agent_state_dict = None
        optimizer_state_dict = None

    sampler = GpuSampler(
        EnvCls=gym_make,
        env_kwargs=env_cfg,
        eval_env_kwargs=env_cfg,
        batch_T=25,  # each sample collect 1 env steps
        batch_B=num_train_env,  # 2 train envs
        max_decorrelation_steps=0,
        eval_n_envs=num_eval_env,
        eval_max_steps=250,
        eval_max_trajectories=10,
    )

    # sampler = SerialSampler(
    #     EnvCls=gym_make,
    #     env_kwargs=env_cfg,
    #     eval_env_kwargs=env_cfg,
    #     batch_T=4,  # each sample collect 1 env steps
    #     batch_B=1,  # 1 train envs
    #     max_decorrelation_steps=0,
    #     eval_n_envs=1,  # 2 eval envs
    #     eval_max_steps=200,  # 10 episodes in total (5 per eval envs)
    # )

    algo = DQN(
        double_dqn=True,
        delta_clip=None,
        learning_rate=1e-3,
        discount=0.99,
        batch_size=64,
        min_steps_learn=int(1e3),
        replay_size=int(1e4),
        replay_ratio=8,
        # 320 * batch_size / replay_ratio = 2560 env steps.
        target_update_interval=2560,
        ReplayBufferCls=UniformReplayBuffer,
        eps_steps=int(5e5),
        initial_optim_state_dict=optimizer_state_dict,
        eval_only=eval_only
    )

    model_kwargs = dict(
        base_only=False
    )

    agent = RelMoGenDqnAgent(
        ModelCls=GibsonDqnModel,
        eps_init=0.5,
        eps_final=0.04,
        eps_eval=0.0,
        initial_model_state_dict=agent_state_dict,
        model_kwargs=model_kwargs
    )

    # agent = GridDqnAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=n_steps,
        log_interval_steps=int(1e4),
        affinity=affinity,
        seed=0,
    )

    game = 'relmogen'
    config = dict(game=game)
    name = 'dqn_' + game
    log_dir = 'relmogen'
    with logger_context(log_dir, run_ID, name, config,
                        snapshot_mode=snapshot_mode,
                        use_summary_writer=True):
        runner.train()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--run_ID', help='run identifier (logging)', type=str, default="0")
    parser.add_argument('--arena', help='which arena to train',
                        type=str, default='push_door')
    parser.add_argument('--cuda_idx', help='gpu to use',
                        type=int, default=None)
    parser.add_argument('--model_path', help='path to the saved model',
                        type=str, default=None)
    parser.add_argument('--eval_only', help='whether to only run evaluation',
                        action='store_true')
    args = parser.parse_args()
    build_and_train(
        run_ID=args.run_ID,
        arena=args.arena,
        cuda_idx=args.cuda_idx,
        model_path=args.model_path,
        eval_only=args.eval_only,
    )
