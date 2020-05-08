
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import make_grid as gym_make
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.grid.grid_dqn_agent import GridDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context


def build_and_train(run_ID=0, cuda_idx=None):
    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=dict(grid_size=4, max_steps=1, obs_inflation=32),
        eval_env_kwargs=dict(grid_size=4, max_steps=1, obs_inflation=32),
        batch_T=1,  # One time-step per sampler iteration.
        batch_B=256,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=1,
        eval_max_steps=100,
        eval_max_trajectories=50,
    )

    algo = DQN(replay_size=int(1e6), batch_size=4096)  # Run with defaults.
    agent = GridDqnAgent()
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
    )

    game = 'grid'
    config = dict(game=game)
    name = "dqn_" + game
    log_dir = "grid"
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
