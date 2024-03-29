
import multiprocessing as mp
from functools import wraps

from rlpyt.agents.base import AgentInputs, CONAgentInputs
from rlpyt.samplers.parallel.base import ParallelSamplerBase
from rlpyt.samplers.parallel.gpu.action_server import ActionServer, CONActionServer
from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector, CONGpuResetCollector,
                                                    GpuEvalCollector)
from rlpyt.utils.collections import namedarraytuple, AttrDict
from rlpyt.utils.synchronize import drain_queue
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer
from rlpyt.envs.base import EnvSpaces
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper


StepBuffer = namedarraytuple("StepBuffer",
                             ["observation", "action", "reward", "done", "agent_info"])
CONStepBuffer = namedarraytuple("StepBuffer",
                             ["observation", "action", "reward", "done", "agent_info", "feature"])


class GpuSamplerBase(ParallelSamplerBase):
    """Base class for parallel samplers which use worker processes to execute
    environment steps on CPU resources but the master process to execute agent
    forward passes for action selection, presumably on GPU.  Use GPU-based
    collecter classes.

    In addition to the usual batch buffer for data samples, allocates a step
    buffer over shared memory, which is used for communication with workers.
    The step buffer includes `observations`, which the workers write and the
    master reads, and `actions`, which the master write and the workers read.
    (The step buffer has leading dimension [`batch_B`], for the number of
    parallel environments, and each worker gets its own slice along that
    dimension.)  The step buffer object is held in both numpy array and torch
    tensor forms over the same memory; e.g. workers write to the numpy array
    form, and the agent is able to read the torch tensor form.

    (Possibly more information about how the stepping works, but write
    in action-server or smwr like that.)
    """

    gpu = True

    def __init__(self, *args, CollectorCls=GpuResetCollector,
                 eval_CollectorCls=GpuEvalCollector, **kwargs):
        # e.g. or use GpuWaitResetCollector, etc...
        super().__init__(*args, CollectorCls=CollectorCls,
                         eval_CollectorCls=eval_CollectorCls, **kwargs)

    def obtain_samples(self, itr):
        """Signals worker to begin environment step execution loop, and drops
        into ``serve_actions()`` method to provide actions to workers based on
        the new observations at each step.
        """
        # self.samples_np[:] = 0  # Reset all batch sample values (optional).
        self.agent.sample_mode(itr)
        self.ctrl.barrier_in.wait()
        self.serve_actions(itr)  # Worker step environments here.
        self.ctrl.barrier_out.wait()
        traj_infos = drain_queue(self.traj_infos_queue)
        return self.samples_pyt, traj_infos

    def evaluate_agent(self, itr):
        """Signals workers to begin agent evaluation loop, and drops into
        ``serve_actions_evaluation()`` to provide actions to workers at each
        step.
        """
        self.ctrl.do_eval.value = True
        self.sync.stop_eval.value = False
        self.agent.eval_mode(itr)
        self.ctrl.barrier_in.wait()
        traj_infos = self.serve_actions_evaluation(itr)
        self.ctrl.barrier_out.wait()
        traj_infos.extend(drain_queue(self.eval_traj_infos_queue,
                                      n_sentinel=self.eval_n_envs))  # Block until all finish submitting.
        self.ctrl.do_eval.value = False
        return traj_infos

    def _agent_init(self, agent, env_cls, env_kwargs,
                    global_B=1, env_ranks=None, env_spaces=None):
        """Initializes the agent, having it *not* share memory, because all
        agent functions (training and sampling) happen in the master process,
        presumably on GPU."""

        if env_spaces is None:
            def target(env_cls, env_kwargs, storage):
                if 'mode' in env_kwargs:
                    env_kwargs['mode'] = 'headless'
                env = env_cls(**env_kwargs)
                storage['observation_space'] = env.env.observation_space
                storage['action_space'] = env.env.action_space
                env.close()

            mgr = mp.Manager()
            storage = mgr.dict()
            w = mp.Process(target=target, args=(env_cls, env_kwargs, storage))
            w.start()
            w.join()
            env_spaces = EnvSpaces(
                observation=GymSpaceWrapper(
                    space=storage['observation_space'],
                    name="obs",
                    null_value=0,
                    force_float32=True,
                ),
                action=GymSpaceWrapper(
                    space=storage['action_space'],
                    name="act",
                    null_value=0,
                    force_float32=True,
                )
            )

        agent.initialize(env_spaces, share_memory=False,
                         global_B=global_B, env_ranks=env_ranks)
        self.agent = agent

    def _build_buffers(self, *args, **kwargs):
        examples = super()._build_buffers(*args, **kwargs)
        self.step_buffer_pyt, self.step_buffer_np = build_step_buffer(
            examples, self.batch_spec.B)
        self.agent_inputs = AgentInputs(self.step_buffer_pyt.observation,
                                        self.step_buffer_pyt.action, self.step_buffer_pyt.reward)
        if self.eval_n_envs > 0:
            self.eval_step_buffer_pyt, self.eval_step_buffer_np = \
                build_step_buffer(examples, self.eval_n_envs)
            self.eval_agent_inputs = AgentInputs(
                self.eval_step_buffer_pyt.observation,
                self.eval_step_buffer_pyt.action,
                self.eval_step_buffer_pyt.reward,
            )
        return examples

    def _build_parallel_ctrl(self, n_worker):
        super()._build_parallel_ctrl(n_worker)
        self.sync.obs_ready = [mp.Semaphore(0) for _ in range(n_worker)]
        self.sync.act_ready = [mp.Semaphore(0) for _ in range(n_worker)]

    def _assemble_common_kwargs(self, *args, **kwargs):
        common_kwargs = super()._assemble_common_kwargs(*args, **kwargs)
        common_kwargs["agent"] = None  # Remove.
        return common_kwargs

    def _assemble_workers_kwargs(self, affinity, seed, n_envs_list):
        workers_kwargs = super()._assemble_workers_kwargs(affinity, seed, n_envs_list)
        i_env = 0
        rank_with_eval = 0
        for rank, w_kwargs in enumerate(workers_kwargs):
            n_envs, eval_n_envs = n_envs_list[rank]
            slice_B = slice(i_env, i_env + n_envs)
            w_kwargs["sync"] = AttrDict(
                stop_eval=self.sync.stop_eval,
                obs_ready=self.sync.obs_ready[rank],
                act_ready=self.sync.act_ready[rank],
            )
            w_kwargs["step_buffer_np"] = self.step_buffer_np[slice_B]
            if eval_n_envs > 0:
                eval_slice_B = slice(self.eval_n_envs_per * rank_with_eval,
                                     self.eval_n_envs_per * (rank_with_eval + 1))
                w_kwargs["eval_step_buffer_np"] = \
                    self.eval_step_buffer_np[eval_slice_B]
                rank_with_eval += 1
            i_env += n_envs
        return workers_kwargs

class GpuSampler(ActionServer, GpuSamplerBase):
    pass

class CONGpuSamplerBase(ParallelSamplerBase):
    """Base class for parallel samplers which use worker processes to execute
    environment steps on CPU resources but the master process to execute agent
    forward passes for action selection, presumably on GPU.  Use GPU-based
    collecter classes.

    In addition to the usual batch buffer for data samples, allocates a step
    buffer over shared memory, which is used for communication with workers.
    The step buffer includes `observations`, which the workers write and the
    master reads, and `actions`, which the master write and the workers read.
    (The step buffer has leading dimension [`batch_B`], for the number of
    parallel environments, and each worker gets its own slice along that
    dimension.)  The step buffer object is held in both numpy array and torch
    tensor forms over the same memory; e.g. workers write to the numpy array
    form, and the agent is able to read the torch tensor form.

    (Possibly more information about how the stepping works, but write
    in action-server or smwr like that.)
    """

    gpu = True

    def __init__(self, *args, CollectorCls=CONGpuResetCollector,
                 eval_CollectorCls=GpuEvalCollector, **kwargs):
        # e.g. or use GpuWaitResetCollector, etc...
        super().__init__(*args, CollectorCls=CollectorCls,
                         eval_CollectorCls=eval_CollectorCls, **kwargs)

    def obtain_samples(self, itr):
        """Signals worker to begin environment step execution loop, and drops
        into ``serve_actions()`` method to provide actions to workers based on
        the new observations at each step.
        """
        # self.samples_np[:] = 0  # Reset all batch sample values (optional).
        self.agent.sample_mode(itr)
        self.ctrl.barrier_in.wait()
        self.serve_actions(itr)  # Worker step environments here.
        self.ctrl.barrier_out.wait()
        traj_infos = drain_queue(self.traj_infos_queue)
        return self.samples_pyt, traj_infos

    def evaluate_agent(self, itr):
        """Signals workers to begin agent evaluation loop, and drops into
        ``serve_actions_evaluation()`` to provide actions to workers at each
        step.
        """
        self.ctrl.do_eval.value = True
        self.sync.stop_eval.value = False
        self.agent.eval_mode(itr)
        self.ctrl.barrier_in.wait()
        traj_infos = self.serve_actions_evaluation(itr)
        self.ctrl.barrier_out.wait()
        traj_infos.extend(drain_queue(self.eval_traj_infos_queue,
                                      n_sentinel=self.eval_n_envs))  # Block until all finish submitting.
        self.ctrl.do_eval.value = False
        return traj_infos

    def _agent_init(self, agent, env_cls, env_kwargs,
                    global_B=1, env_ranks=None, env_spaces=None):
        """Initializes the agent, having it *not* share memory, because all
        agent functions (training and sampling) happen in the master process,
        presumably on GPU."""
        #
        if env_spaces is None:
            def target(env_cls, env_kwargs, storage):
                if 'mode' in env_kwargs:
                    env_kwargs['mode'] = 'headless'
                env = env_cls(**env_kwargs)
                storage['observation_space'] = env.env.observation_space
                storage['action_space'] = env.env.action_space
                env.close()

            mgr = mp.Manager()
            storage = mgr.dict()
            w = mp.Process(target=target, args=(env_cls, env_kwargs, storage))
            w.start()
            w.join()
            env_spaces = EnvSpaces(
                observation=GymSpaceWrapper(
                    space=storage['observation_space'],
                    name="obs",
                    null_value=0,
                    force_float32=True,
                ),
                action=GymSpaceWrapper(
                    space=storage['action_space'],
                    name="act",
                    null_value=0,
                    force_float32=True,
                )
            )

        agent.initialize(env_spaces, share_memory=False,
                         global_B=global_B, env_ranks=env_ranks)
        self.agent = agent

    def _build_buffers(self, *args, **kwargs):
        examples = super()._build_buffers(*args, **kwargs)
        self.step_buffer_pyt, self.step_buffer_np = CON_build_step_buffer(
            examples, self.batch_spec.B)
        self.agent_inputs = CONAgentInputs(self.step_buffer_pyt.observation,
                                        self.step_buffer_pyt.feature)
        if self.eval_n_envs > 0:
            self.eval_step_buffer_pyt, self.eval_step_buffer_np = \
                CON_build_step_buffer(examples, self.eval_n_envs)
            self.eval_agent_inputs = CONAgentInputs(
                self.eval_step_buffer_pyt.observation,
                self.eval_step_buffer_pyt.feature
            )
        return examples

    def _build_parallel_ctrl(self, n_worker):
        super()._build_parallel_ctrl(n_worker)
        self.sync.obs_ready = [mp.Semaphore(0) for _ in range(n_worker)]
        self.sync.act_ready = [mp.Semaphore(0) for _ in range(n_worker)]

    def _assemble_common_kwargs(self, *args, **kwargs):
        common_kwargs = super()._assemble_common_kwargs(*args, **kwargs)
        common_kwargs["agent"] = None  # Remove.
        return common_kwargs

    def _assemble_workers_kwargs(self, affinity, seed, n_envs_list):
        workers_kwargs = super()._assemble_workers_kwargs(affinity, seed, n_envs_list)
        i_env = 0
        rank_with_eval = 0
        for rank, w_kwargs in enumerate(workers_kwargs):
            n_envs, eval_n_envs = n_envs_list[rank]
            slice_B = slice(i_env, i_env + n_envs)
            w_kwargs["sync"] = AttrDict(
                stop_eval=self.sync.stop_eval,
                obs_ready=self.sync.obs_ready[rank],
                act_ready=self.sync.act_ready[rank],
            )
            w_kwargs["step_buffer_np"] = self.step_buffer_np[slice_B]
            if eval_n_envs > 0:
                eval_slice_B = slice(self.eval_n_envs_per * rank_with_eval,
                                     self.eval_n_envs_per * (rank_with_eval + 1))
                w_kwargs["eval_step_buffer_np"] = \
                    self.eval_step_buffer_np[eval_slice_B]
                rank_with_eval += 1
            i_env += n_envs
        return workers_kwargs

class CONGpuSampler(CONActionServer, CONGpuSamplerBase):
    pass


def build_step_buffer(examples, B):
    step_bufs = {k: buffer_from_example(examples[k], B, share_memory=True)
                 for k in ["observation", "action", "reward", "done", "agent_info"]}
    step_buffer_np = StepBuffer(**step_bufs)
    step_buffer_pyt = torchify_buffer(step_buffer_np)
    return step_buffer_pyt, step_buffer_np


def CON_build_step_buffer(examples, B):
    step_bufs = {k: buffer_from_example(examples[k], B, share_memory=True)
                 for k in ["observation", "action", "reward", "done", "agent_info", "feature"]}
    step_buffer_np = CONStepBuffer(**step_bufs)
    step_buffer_pyt = torchify_buffer(step_buffer_np)
    return step_buffer_pyt, step_buffer_np
