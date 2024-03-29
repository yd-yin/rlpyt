
import psutil
import time
import torch
import numpy as np

from rlpyt.utils.collections import AttrDict
from rlpyt.utils.logging import logger
from rlpyt.utils.seed import set_seed, set_envs_seeds


def initialize_worker(rank, seed=None, cpu=None, torch_threads=None):
    """Assign CPU affinity, set random seed, set torch_threads if needed to
    prevent MKL deadlock.
    """
    log_str = f"Sampler rank {rank} initialized"
    cpu = [cpu] if isinstance(cpu, int) else cpu
    p = psutil.Process()
    try:
        if cpu is not None:
            p.cpu_affinity(cpu)
        cpu_affin = p.cpu_affinity()
    except AttributeError:
        cpu_affin = "UNAVAILABLE MacOS"
    log_str += f", CPU affinity {cpu_affin}"
    torch_threads = (1 if torch_threads is None and cpu is not None else
        torch_threads)  # Default to 1 to avoid possible MKL hang.
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)
    log_str += f", Torch threads {torch.get_num_threads()}"
    if seed is not None:
        set_seed(seed)
        time.sleep(0.3)  # (so the printing from set_seed is not intermixed)
        log_str += f", Seed {seed}"
    logger.log(log_str)


def sampling_process(common_kwargs, worker_kwargs, index=None):
    """Target function used for forking parallel worker processes in the
    samplers. After ``initialize_worker()``, it creates the specified number
    of environment instances and gives them to the collector when
    instantiating it.  It then calls collector startup methods for
    environments and agent.  If applicable, instantiates evaluation
    environment instances and evaluation collector.

    Then enters infinite loop, waiting for signals from master to collect
    training samples or else run evaluation, until signaled to exit.
    """
    c, w = AttrDict(**common_kwargs), AttrDict(**worker_kwargs)
    env_len = len(c.env_kwargs['scene_names'])
    initialize_worker(w.rank, w.seed, w.cpus, c.torch_threads)

    if w.get("n_envs", 0) > 0:
        # print('\n\n\n')
        # print('w.rank', w.rank)         # rank is the index of the subprocess, within range(batch_B)
        # print('w.n_envs', w.n_envs)     # always equals to 1
        # print('env_len', env_len)       # number of input envs
        # print('global_B', c.global_B)   # equals to batch_B
        # print(w.keys())                 # dict_keys(['rank', 'env_ranks', 'seed', 'cpus', 'n_envs', 'eval_n_envs', 'samples_np', 'sync', 'step_buffer_np'])
        # print(c.keys())                 # dict_keys(['EnvCls', 'env_kwargs', 'agent', 'batch_T', 'CollectorCls', 'TrajInfoCls', 'traj_infos_queue', 'ctrl', 'max_decorrelation_steps', 'torch_threads', 'global_B'])
        # print('\n\n\n')

        # Pass `w.rank` to env creation for training on different scenes
        batch_B = c.global_B
        if env_len <= batch_B:
            scene_idx = w.rank % env_len
        else:
            print('\nThe batch_B is smaller than #training_envs, random select a training_env for each process\n')
            #scene_idx = np.random.randint(env_len)
            scene_idx = index


        envs = [c.EnvCls(**c.env_kwargs, scene_idx=scene_idx, worker_id=w.rank) for _ in range(w.n_envs)]
        # print('length of envs:', len(envs))     # always equals to 1
        # print('Env:', envs[0].scene_dir)        # indicating the scene in current subprocess
        set_envs_seeds(envs, w.seed)
        collector = c.CollectorCls(
            rank=w.rank,
            envs=envs,
            samples_np=w.samples_np,
            batch_T=c.batch_T,
            TrajInfoCls=c.TrajInfoCls,
            agent=c.get("agent", None),  # Optional depending on parallel setup.
            sync=w.get("sync", None),
            step_buffer_np=w.get("step_buffer_np", None),
            global_B=c.get("global_B", 1),
            env_ranks=w.get("env_ranks", None),
        )
        agent_inputs, traj_infos = collector.start_envs(c.max_decorrelation_steps)
        collector.start_agent()
        

    else:
        envs = []
        collector = None

    if w.get("eval_n_envs", 0) > 0:
        eval_envs = [c.EnvCls(**c.eval_env_kwargs) for _ in range(w.eval_n_envs)]
        set_envs_seeds(eval_envs, w.seed)
        eval_collector = c.eval_CollectorCls(
            rank=w.rank,
            envs=eval_envs,
            TrajInfoCls=c.TrajInfoCls,
            traj_infos_queue=c.eval_traj_infos_queue,
            max_T=c.eval_max_T,
            agent=c.get("agent", None),
            sync=w.get("sync", None),
            step_buffer_np=w.get("eval_step_buffer_np", None),
        )

    else:
        eval_envs = list()
        eval_collector = None

    ctrl = c.ctrl
    ctrl.barrier_out.wait()
    while True:
        if collector is not None:
            collector.reset_if_needed(agent_inputs)  # Outside barrier?
        ctrl.barrier_in.wait()
        if ctrl.quit.value:
            break
        if ctrl.do_eval.value:
            if eval_collector is not None:
                eval_collector.collect_evaluation(ctrl.itr.value)  # Traj_infos to queue inside.
        else:
            if collector is not None:
                agent_inputs, traj_infos, completed_infos = collector.collect_batch(
                    agent_inputs, traj_infos, ctrl.itr.value)
                for info in completed_infos:
                    c.traj_infos_queue.put(info)
        ctrl.barrier_out.wait()

    for env in envs + eval_envs:
        env.close()
