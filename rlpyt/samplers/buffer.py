
import multiprocessing as mp
import numpy as np
from IPython import embed
from collections import namedtuple

from rlpyt.utils.buffer import buffer_from_example, torchify_buffer
from rlpyt.agents.base import AgentInputs, CONAgentInputs
from rlpyt.samplers.collections import (Samples, AgentSamples, AgentSamplesBsv,
                                        EnvSamples, CONAgentSamples, CONAgentSamplesBsv,
                                        CONEnvSamples)


def build_samples_buffer(agent, env_cls, env_kwargs, batch_spec,
                         bootstrap_value=False, agent_shared=True,
                         env_shared=True, subprocess=True, examples=None):
    """Recommended to step/reset agent and env in subprocess, so it doesn't
    affect settings in master before forking workers (e.g. torch num_threads
    (MKL) may be set at first forward computation.)"""
    if hasattr(agent, 'checkCON'):
        get_example_function = CON_get_example_outputs
    else:
        get_example_function = get_example_outputs
    if examples is None:
        if subprocess:
            mgr = mp.Manager()
            examples = mgr.dict()  # Examples pickled back to master.
            w = mp.Process(target=get_example_function,
                           args=(agent, env_cls, env_kwargs,
                                 examples, subprocess))
            w.start()
            w.join()
            examples = dict(examples)
            examples["env_info"] = namedtuple("info", list(examples["env_info"]))(
                succ=False,
                cam_succ=False,
                timeout=False,
                outbounds=False,
                getnan=False,
            )
        else:
            examples = dict()
            get_example_function(agent, env_cls, env_kwargs,
                                examples, subprocess)


    if hasattr(agent, 'checkCON'):
        T, B = batch_spec
        all_action = buffer_from_example(
            examples["action"], (T + 1, B), agent_shared)
        action = all_action[1:]
        # Writing to action will populate prev_action.
        agent_info = buffer_from_example(
            examples["agent_info"], (T, B), agent_shared)
        feature = buffer_from_example(
            examples["feature"], (T, B), agent_shared)
        agent_buffer = CONAgentSamples(
            action=action,
            agent_info=agent_info,
            feature=feature,
        )
        if bootstrap_value:
            bv = buffer_from_example(
                examples["agent_info"].value, (1, B), agent_shared)
            agent_buffer = CONAgentSamplesBsv(*agent_buffer, bootstrap_value=bv)

        observation = buffer_from_example(
            examples["observation"], (T, B), env_shared)
        all_reward = buffer_from_example(
            examples["reward"], (T + 1, B), env_shared)
        reward = all_reward[1:]
        # Writing to reward will populate prev_reward.
        done = buffer_from_example(examples["done"], (T, B), env_shared)
        env_info = buffer_from_example(examples["env_info"], (T, B), env_shared)
        env_buffer = CONEnvSamples(
            observation=observation,
            reward=reward,
            done=done,
            env_info=env_info,
        )
        samples_np = Samples(agent=agent_buffer, env=env_buffer)
        samples_pyt = torchify_buffer(samples_np)
        return samples_pyt, samples_np, examples
       


    T, B = batch_spec
    all_action = buffer_from_example(
        examples["action"], (T + 1, B), agent_shared)
    action = all_action[1:]
    # Writing to action will populate prev_action.
    prev_action = all_action[:-1]
    agent_info = buffer_from_example(
        examples["agent_info"], (T, B), agent_shared)
    agent_buffer = AgentSamples(
        action=action,
        prev_action=prev_action,
        agent_info=agent_info,
    )
    if bootstrap_value:
        bv = buffer_from_example(
            examples["agent_info"].value, (1, B), agent_shared)
        agent_buffer = AgentSamplesBsv(*agent_buffer, bootstrap_value=bv)

    observation = buffer_from_example(
        examples["observation"], (T, B), env_shared)
    all_reward = buffer_from_example(
        examples["reward"], (T + 1, B), env_shared)
    reward = all_reward[1:]
    # Writing to reward will populate prev_reward.
    prev_reward = all_reward[:-1]
    done = buffer_from_example(examples["done"], (T, B), env_shared)
    env_info = buffer_from_example(examples["env_info"], (T, B), env_shared)
    env_buffer = EnvSamples(
        observation=observation,
        reward=reward,
        prev_reward=prev_reward,
        done=done,
        env_info=env_info,
    )
    samples_np = Samples(agent=agent_buffer, env=env_buffer)
    samples_pyt = torchify_buffer(samples_np)
    return samples_pyt, samples_np, examples



def get_example_outputs(agent, env_cls, env_kwargs, examples, subprocess=False):
    """Do this in a sub-process to avoid setup conflict in master/workers (e.g.
    MKL)."""
    if subprocess:  # i.e. in subprocess.
        import torch
        torch.set_num_threads(1)  # Some fix to prevent MKL hang.

    if 'mode' in env_kwargs:
        env_kwargs['mode'] = 'headless'
    env = env_cls(**env_kwargs)

    o = env.reset()
    a = env.action_space.sample()
    o, r, d, env_info = env.step(a)
    env.close()

    # Must match torch float dtype here.
    r = np.asarray(r, dtype="float32")
    agent.reset()
    scene_names = env_kwargs['scene_names']
    seq_names = env_kwargs['seq_names']
    agent_inputs = torchify_buffer(AgentInputs(o, a, r))
    a, agent_info = agent.step(*agent_inputs)
    if "prev_rnn_state" in agent_info:
        # Agent leaves B dimension in, strip it: [B,N,H] --> [N,H]
        agent_info = agent_info._replace(
            prev_rnn_state=agent_info.prev_rnn_state[0])
    examples["observation"] = o
    examples["reward"] = r
    examples["done"] = d
    examples["env_info"] = env_info._fields
    examples["action"] = a  # OK to put torch tensor here, could numpify.
    examples["agent_info"] = agent_info


def CON_get_example_outputs(agent, env_cls, env_kwargs, examples, subprocess=False):
    """Do this in a sub-process to avoid setup conflict in master/workers (e.g.
    MKL)."""
    if subprocess:  # i.e. in subprocess.
        import torch
        torch.set_num_threads(1)  # Some fix to prevent MKL hang.

    if 'mode' in env_kwargs:
        env_kwargs['mode'] = 'headless'
    env = env_cls(**env_kwargs)

    o = env.reset()
    a = env.action_space.sample()
    o, r, d, env_info = env.step(a)
    env.close()
    feature = env.getFeature()



    # Must match torch float dtype here.
    r = np.asarray(r, dtype="float32")
    agent.reset()
    
    if hasattr(agent, 'set_agent_extra_info'):
        agent.set_agent_extra_info(1, scene_names, seq_names)
    agent_inputs = torchify_buffer(CONAgentInputs(o, np.expand_dims(feature, axis=0)))
    a, agent_info, feature = agent.step(*agent_inputs)
    if "prev_rnn_state" in agent_info:
        # Agent leaves B dimension in, strip it: [B,N,H] --> [N,H]
        agent_info = agent_info._replace(
            prev_rnn_state=agent_info.prev_rnn_state[0])
    examples["observation"] = o
    examples["reward"] = r
    examples["done"] = d
    examples["env_info"] = env_info._fields
    examples["action"] = a  # OK to put torch tensor here, could numpify.
    examples["agent_info"] = agent_info
    examples["feature"] = feature
