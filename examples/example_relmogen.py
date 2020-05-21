
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
from rlpyt.envs.gym_wrapper import make_relmogen as gym_make
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.relmogen.relmogen_dqn_agent import RelMoGenDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context


def build_and_train(run_ID=0, cuda_idx=None):
    gibson_cfg = os.path.join(os.path.dirname(rlpyt.__file__), 'envs/fetch_p2p_nav.yaml')
    print(gibson_cfg)

    env_cfg = dict(
        config_file=gibson_cfg,
        action_timestep=1.0 / 24.0,
        physics_timestep=1.0 / 240.0,
        device_idx=1,
        downsize_ratio=8,
    )
    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=env_cfg,
        eval_env_kwargs=env_cfg,
        batch_T=4,  # each sample collect 4 env steps
        batch_B=8,  # 8 train envs
        max_decorrelation_steps=0,
        eval_n_envs=2,  # 2 eval envs
        eval_max_steps=200,  # 10 episodes in total (5 per eval envs)
    )

    algo = DQN(
        replay_size=int(1e5),
        batch_size=256,
        min_steps_learn=int(5e3),
        target_update_interval=39, # 39 * 256 = 1e4 env steps.
    )

    agent = GridDqnAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e4,
        affinity=dict(cuda_idx=cuda_idx),
    )

    game = 'relmogen'
    config = dict(game=game)
    name = "dqn_" + game
    log_dir = "relmogen"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last", use_summary_writer=True):
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
