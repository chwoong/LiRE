import numpy as np
import torch
import torch.nn.functional as F
import gym
import os

import dmc2gym
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

from gym.wrappers.time_limit import TimeLimit
from rlkit.envs.wrappers import NormalizedBoxEnv
import pickle as pkl


def make_metaworld_env(env_name, seed):
    env_name = env_name.replace("metaworld_", "")
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]

    env = env_cls()
    # print("partially observe", env._partially_observable) Ture
    # print("env._freeze_rand_vec", env._freeze_rand_vec) True
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)
    return TimeLimit(NormalizedBoxEnv(env), env.max_path_length)


def make_dmc_env(env_name, seed):
    env_name = env_name.replace("dmc_", "")
    domain_name, task_name = env_name.split("-")
    domain_name = domain_name.lower()
    task_name = task_name.lower()
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
    )
    return env


def MetaWorld_dataset(config):
    """
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if config.human == False:
        base_path = os.path.join(os.getcwd(), "dataset/MetaWorld/")
        env_name = config.env
        base_path += str(env_name.replace("metaworld_", ""))
        dataset = dict()
        for seed in range(3):
            path = base_path + f"/saved_replay_buffer_1000000_seed{seed}.pkl"
            with open(path, "rb") as f:
                load_dataset = pkl.load(f)

            for key in load_dataset.keys():
                load_dataset[key] = load_dataset[key][
                    : int(config.data_quality * 100_000)
                ]
            load_dataset["terminals"] = load_dataset["dones"][
                : int(config.data_quality * 100_000)
            ]
            load_dataset.pop("dones", None)

            for key in load_dataset.keys():
                if key not in dataset:
                    dataset[key] = load_dataset[key]
                else:
                    dataset[key] = np.concatenate(
                        (dataset[key], load_dataset[key]), axis=0
                    )
    elif config.human == True:
        base_path = os.path.join(os.getcwd(), "human_feedback/")
        base_path += f"{config.env}/dataset.pkl"
        with open(base_path, "rb") as f:
            dataset = pkl.load(f)
            dataset["observations"] = np.array(dataset["observations"])
            dataset["actions"] = np.array(dataset["actions"])
            dataset["next_observations"] = np.array(dataset["next_observations"])
            dataset["rewards"] = np.array(dataset["rewards"])
            dataset["terminals"] = np.array(dataset["dones"])

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    dataset["rewards"] = dataset["rewards"].reshape(-1)
    dataset["terminals"] = dataset["terminals"].reshape(-1)

    for i in range(N):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["next_observations"][i].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }


def DMC_dataset(config):
    base_path = os.path.join(os.getcwd(), "dataset/DMControl/")
    env_name = config.env.replace("dmc_", "")
    base_path += str(env_name)
    dataset = dict()
    for seed in range(3):
        path = base_path + f"/saved_replay_buffer_1000000_seed{seed}.pkl"
        with open(path, "rb") as f:
            load_dataset = pkl.load(f)

        if "humanoid" in env_name:
            for key in load_dataset.keys():
                load_dataset[key] = load_dataset[key][
                    200000 : int(config.data_quality * 100_000)
                ]
            load_dataset["terminals"] = load_dataset["dones"][
                0 : int(config.data_quality * 100_000) - 200000
            ]
            load_dataset.pop("dones", None)
        else:
            for key in load_dataset.keys():
                load_dataset[key] = load_dataset[key][
                    0 : int(config.data_quality * 100_000)
                ]
            load_dataset["terminals"] = load_dataset["dones"][
                0 : int(config.data_quality * 100_000) - 0
            ]
            load_dataset.pop("dones", None)

        for key in load_dataset.keys():
            if key not in dataset:
                dataset[key] = load_dataset[key]
            else:
                dataset[key] = np.concatenate((dataset[key], load_dataset[key]), axis=0)
        # print("shape", load_dataset["rewards"].shape, "from seed ", seed, end=",  ")
    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    dataset["rewards"] = dataset["rewards"].reshape(-1)
    dataset["terminals"] = dataset["terminals"].reshape(-1)

    for i in range(N):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["next_observations"][i].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }
