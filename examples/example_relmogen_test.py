
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""

import os

import rlpyt
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.envs.gym_wrapper import make_relmogen as gym_make
# from rlpyt.envs.gym_wrapper import make_grid as gym_make
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.grid.grid_dqn_agent import GridDqnAgent
from rlpyt.agents.dqn.relmogen.relmogen_dqn_agent import RelMoGenDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.replays.non_sequence.uniform import UniformReplayBuffer


def build_and_train(run_ID=0, cuda_idx=None):
    gibson_cfg = os.path.join(os.path.dirname(rlpyt.__file__), 'envs/fetch_p2p_nav.yaml')
    print(gibson_cfg)

    env_cfg = dict(
        mode='headless',
        config_file=gibson_cfg,
        action_timestep=1.0 / 24.0,
        physics_timestep=1.0 / 240.0,
        device_idx=0,
        downsize_ratio=8,
    )
    # env_cfg = dict(grid_size=4, max_steps=1, obs_inflation=32)

    affinity = dict(
        cuda_idx=cuda_idx,
        workers_cpus=[39, 9, 10, 11, 12, 13, 14, 15, 18, 19, 29, 30],
    )

    sampler = GpuSampler(
        EnvCls=gym_make,
        env_kwargs=env_cfg,
        eval_env_kwargs=env_cfg,
        batch_T=4,  # each sample collect 1 env steps
        batch_B=8,  # 2 train envs
        max_decorrelation_steps=0,
        eval_n_envs=4,
        eval_max_steps=600,
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
        batch_size=256,
        min_steps_learn=int(5e3),
        replay_size=int(1.5e5),
        replay_ratio=8,  # replay_ratio = batch_size / (batch_T * batch_B)
        target_update_interval=156,  # 250 * (batch_size / replay_ratio = 4) = 1e3 env steps.
        ReplayBufferCls=UniformReplayBuffer,
        eps_steps=int(5e5),
    )

    agent = RelMoGenDqnAgent()
    # agent = GridDqnAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=int(1e6),
        log_interval_steps=5e3,
        affinity=affinity,
    )

    game = 'relmogen'
    config = dict(game=game)
    name = "dqn_" + game
    log_dir = "relmogen"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="all", use_summary_writer=True):
        runner.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    args = parser.parse_args()
    build_and_train(
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
