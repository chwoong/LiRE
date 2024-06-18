import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gym, random, torch, os, uuid
import rich
import wandb


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def get_indices(traj_total, config):
    traj_idx = np.random.choice(traj_total, replace=False)
    idx_st = 500 * traj_idx + np.random.randint(0, 500 - config.segment_size)
    idx = [[j for j in range(idx_st, idx_st + config.segment_size)]]
    return idx


def consist_test_dataset(
    dataset, test_feedback_num, traj_total, segment_size, threshold
):
    test_traj_idx = np.random.choice(traj_total, 2 * test_feedback_num, replace=True)
    test_idx = [
        500 * i + np.random.randint(0, 500 - segment_size) for i in test_traj_idx
    ]
    test_idx_st_1 = test_idx[:test_feedback_num]
    test_idx_st_2 = test_idx[test_feedback_num:]
    test_idx_1 = [[j for j in range(i, i + segment_size)] for i in test_idx_st_1]
    test_idx_2 = [[j for j in range(i, i + segment_size)] for i in test_idx_st_2]
    test_labels = obtain_labels(
        dataset,
        test_idx_1,
        test_idx_2,
        segment_size=segment_size,
        threshold=threshold,
        noise=0.0,
    )
    test_binary_labels = obtain_labels(
        dataset,
        test_idx_1,
        test_idx_2,
        segment_size=segment_size,
        threshold=0,
        noise=0.0,
    )
    test_obs_act_1 = np.concatenate(
        (dataset["observations"][test_idx_1], dataset["actions"][test_idx_1]),
        axis=-1,
    )
    test_obs_act_2 = np.concatenate(
        (dataset["observations"][test_idx_2], dataset["actions"][test_idx_2]),
        axis=-1,
    )
    return test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels


def collect_feedback(dataset, traj_total, config):
    multiple_ranked_list = []
    print("config.feedback_num", config.feedback_num)
    if config.feedback_type == "RLT":
        # If q_budget = 1, we collect independent pairwise feedback
        print("Construct RLT")
        cum_query = 0
        current_query_num = 0
        single_ranked_list = []
        while cum_query < config.feedback_num:
            if current_query_num >= config.q_budget:
                multiple_ranked_list.append(single_ranked_list)
                current_query_num = 0
                single_ranked_list = []
            # binary search
            group = []
            if len(single_ranked_list) == 0:
                idx_1 = get_indices(traj_total, config)
                idx_2 = get_indices(traj_total, config)
                idx_st_1 = idx_1[0][0]
                idx_st_2 = idx_2[0][0]
                reward_1 = np.round(np.sum(dataset["rewards"][idx_1], axis=1), 2).item()
                reward_2 = np.round(np.sum(dataset["rewards"][idx_2], axis=1), 2).item()
                label = obtain_labels(
                    dataset,
                    idx_1,
                    idx_2,
                    segment_size=config.segment_size,
                    threshold=config.threshold,
                    noise=config.noise,
                )
                cum_query += 1
                current_query_num += 1
                if np.all(label[0] == [1, 0]):
                    group_1, group_2 = [(idx_st_1, reward_1)], [(idx_st_2, reward_2)]
                    single_ranked_list.append(group_2)
                    single_ranked_list.append(group_1)
                elif np.all(label[0] == [0.5, 0.5]):
                    group = [(idx_st_1, reward_1), (idx_st_2, reward_2)]
                    single_ranked_list.append(group)
                elif np.all(label[0] == [0, 1]):
                    group_1, group_2 = [(idx_st_1, reward_1)], [(idx_st_2, reward_2)]
                    single_ranked_list.append(group_1)
                    single_ranked_list.append(group_2)
            else:
                idx = get_indices(traj_total, config)
                idx_st = idx[0][0]
                reward = np.round(np.sum(dataset["rewards"][idx], axis=1), 2).item()
                low = 0
                high = len(single_ranked_list)
                pos = 0
                insert = False
                while low < high:
                    mid = (low + high) // 2
                    mid_num = (low + high) // 2
                    mid = (mid + mid_num) // 2
                    len_mid = len(single_ranked_list[mid])
                    # random select one element in the mid group
                    random_idx = np.random.randint(0, len_mid)
                    compare_idx = single_ranked_list[mid][random_idx][0]
                    compare_seg_idx = [
                        [
                            j
                            for j in range(
                                compare_idx, compare_idx + config.segment_size
                            )
                        ]
                    ]
                    label = obtain_labels(
                        dataset,
                        idx,
                        compare_seg_idx,
                        segment_size=config.segment_size,
                        threshold=config.threshold,
                        noise=config.noise,
                    )
                    cum_query += 1
                    current_query_num += 1
                    if np.all(label[0] == [0, 1]):
                        high = mid
                        pos = mid
                    elif np.all(label[0] == [1, 0]):
                        low = mid + 1
                        pos = mid + 1
                    else:
                        if cum_query <= config.feedback_num:
                            single_ranked_list[mid].append((idx_st, reward))
                        insert = True
                        break
                if insert == False and cum_query <= config.feedback_num:
                    single_ranked_list.insert(pos, [(idx_st, reward)])
        multiple_ranked_list.append(single_ranked_list)
        return multiple_ranked_list
    elif config.feedback_type == "SeqRank":
        print("Sequential Pairwise feedback (SeqRank)")
        single_ranked_list = []
        up = 0
        if len(single_ranked_list) == 0:
            idx_1 = get_indices(traj_total, config)
            idx_2 = get_indices(traj_total, config)
            idx_st_1 = idx_1[0][0]
            idx_st_2 = idx_2[0][0]
            reward_1 = np.round(np.sum(dataset["rewards"][idx_1], axis=1), 2).item()
            reward_2 = np.round(np.sum(dataset["rewards"][idx_2], axis=1), 2).item()
            label = obtain_labels(
                dataset,
                idx_1,
                idx_2,
                segment_size=config.segment_size,
                threshold=config.threshold,
                noise=config.noise,
            )
            if np.all(label[0] == [1, 0]):
                group_1, group_2 = [(idx_st_1, reward_1)], [(idx_st_2, reward_2)]
                single_ranked_list.append(group_2)
                single_ranked_list.append(group_1)
                up = -1
            elif np.all(label[0] == [0.5, 0.5]):
                group = [(idx_st_1, reward_1), (idx_st_2, reward_2)]
                single_ranked_list.append(group)
                up = 0
            elif np.all(label[0] == [0, 1]):
                group_1, group_2 = [(idx_st_1, reward_1)], [(idx_st_2, reward_2)]
                single_ranked_list.append(group_1)
                single_ranked_list.append(group_2)
                up = 1
        last_idx = idx_2
        last_idx_st = last_idx[0][0]
        for i in range(config.feedback_num - 1):
            idx = get_indices(traj_total, config)
            idx_st = idx[0][0]
            reward_last = np.round(
                np.sum(dataset["rewards"][last_idx], axis=1), 2
            ).item()
            reward = np.round(np.sum(dataset["rewards"][idx], axis=1), 2).item()
            label = obtain_labels(
                dataset,
                last_idx,
                idx,
                segment_size=config.segment_size,
                threshold=config.threshold,
                noise=config.noise,
            )
            if np.all(label[0] == [1, 0]):
                group_1, group_2 = [(last_idx_st, reward_last)], [(idx_st, reward)]
                curr_up = -1
                if up == curr_up or up == 0:
                    # insert front of single_ranked_list
                    single_ranked_list.insert(0, group_2)
                    up = -1
                else:
                    multiple_ranked_list.append(single_ranked_list)
                    single_ranked_list = []
                    single_ranked_list.append(group_2)
                    single_ranked_list.append(group_1)
                    up = -1
            elif np.all(label[0] == [0.5, 0.5]):
                if up == -1:
                    single_ranked_list[0].append((idx_st, reward))
                else:
                    single_ranked_list[-1].append((idx_st, reward))
            elif np.all(label[0] == [0, 1]):
                group_1, group_2 = [(last_idx_st, reward_last)], [(idx_st, reward)]
                curr_up = 1
                if up == curr_up or up == 0:
                    single_ranked_list.append(group_2)
                    up = 1
                else:
                    multiple_ranked_list.append(single_ranked_list)
                    single_ranked_list = []
                    single_ranked_list.append(group_1)
                    single_ranked_list.append(group_2)
                    up = 1
            last_idx = idx
            last_idx_st = idx[0][0]
        multiple_ranked_list.append(single_ranked_list)
        return multiple_ranked_list


def collect_human_feedback(dataset, config):
    print("Human feedback")
    multiple_ranked_list = []
    if config.feedback_type == "RLT":
        # indepednent sampling
        if config.q_budget == 1:
            print("Independent pairwise feedback")
            path = f"./human_feedback/{config.env}/_Independent.txt"
            with open(path, "r") as f:
                for line in f:
                    single_ranked_list = []
                    line = line.split(" ")
                    idx_1 = int(line[0])
                    idx_2 = int(line[1])
                    label = int(line[2])
                    reward_1 = np.round(
                        np.sum(dataset["rewards"][idx_1 : idx_1 + config.segment_size]),
                        2,
                    ).item()
                    reward_2 = np.round(
                        np.sum(dataset["rewards"][idx_2 : idx_2 + config.segment_size]),
                        2,
                    ).item()
                    if label == 1:
                        group_1, group_2 = [(idx_1, reward_1)], [(idx_2, reward_2)]
                        single_ranked_list.append(group_2)
                        single_ranked_list.append(group_1)
                    elif label == 2:
                        group = [(idx_1, reward_1), (idx_2, reward_2)]
                        single_ranked_list.append(group)
                    elif label == 3:
                        group_1, group_2 = [(idx_1, reward_1)], [(idx_2, reward_2)]
                        single_ranked_list.append(group_1)
                        single_ranked_list.append(group_2)
                    multiple_ranked_list.append(single_ranked_list)
        # LiRE
        elif config.q_budget > 1:
            print("Construct RLT (LiRE)")
            for s in range(0, 2):
                path = f"./human_feedback/{config.env}/_RLT_{s}.txt"
                single_ranked_list = []
                with open(path, "r") as f:
                    for line in f:
                        line = line.split("[")[1].split("]")[0].split(", ")
                        group = []
                        for idx in line:
                            idx = int(idx)
                            group.append(
                                (
                                    idx,
                                    np.round(
                                        np.sum(
                                            dataset["rewards"][
                                                idx : idx + config.segment_size
                                            ]
                                        ),
                                        2,
                                    ),
                                )
                            )
                        single_ranked_list.append(group)
                multiple_ranked_list.append(single_ranked_list)

    elif config.feedback_type == "SeqRank":
        print("Sequential Pairwise feedback (SeqRank)")
        path = f"./human_feedback/{config.env}/_SeqRank.txt"
        single_ranked_list = []
        idx_st_1 = []
        idx_st_2 = []
        labels = []
        raw_labels = []
        reward_1 = []
        reward_2 = []
        with open(path, "r") as f:
            for line in f:
                line = line.split("\t")
                index_1 = int(line[0])
                index_2 = int(line[1])
                label = int(line[2])
                idx_st_1.append(index_1)
                idx_st_2.append(index_2)
                raw_labels.append(label)
            idx_1 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_1]
            idx_2 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_2]
            reward_1 = np.sum(dataset["rewards"][idx_1], axis=1)
            reward_2 = np.sum(dataset["rewards"][idx_2], axis=1)
            for i in range(len(raw_labels)):
                if raw_labels[i] == 1:
                    labels.append([1, 0])
                elif raw_labels[i] == 2:
                    labels.append([0.5, 0.5])
                elif raw_labels[i] == 3:
                    labels.append([0, 1])
            labels = np.array(labels)
        if np.all(labels[0] == [1, 0]):
            group_1, group_2 = [(idx_st_1[0], reward_1[0])], [
                (idx_st_2[0], reward_2[0])
            ]
            single_ranked_list.append(group_2)
            single_ranked_list.append(group_1)
            up = -1
        elif np.all(labels[0] == [0.5, 0.5]):
            group = [(idx_st_1[0], reward_1[0]), (idx_st_2[0], reward_2[0])]
            single_ranked_list.append(group)
            up = 0
        elif np.all(labels[0] == [0, 1]):
            group_1, group_2 = [(idx_st_1[0], reward_1[0])], [
                (idx_st_2[0], reward_2[0])
            ]
            single_ranked_list.append(group_1)
            single_ranked_list.append(group_2)
            up = 1
        for i in range(1, len(labels)):
            if np.all(labels[i] == [1, 0]):
                group_1, group_2 = [(idx_st_1[i], reward_1[i])], [
                    (idx_st_2[i], reward_2[i])
                ]
                curr_up = -1
                if up == curr_up or up == 0:
                    # insert front of single_ranked_list
                    single_ranked_list.insert(0, group_2)
                    up = -1
                else:
                    multiple_ranked_list.append(single_ranked_list)
                    single_ranked_list = []
                    single_ranked_list.append(group_2)
                    single_ranked_list.append(group_1)
                    up = -1
            elif np.all(labels[i] == [0.5, 0.5]):
                if up == -1:
                    single_ranked_list[0].append((idx_st_2[i], reward_2[i]))
                else:
                    single_ranked_list[-1].append((idx_st_2[i], reward_2[i]))
            elif np.all(labels[i] == [0, 1]):
                group_1, group_2 = [(idx_st_1[i], reward_1[i])], [
                    (idx_st_2[i], reward_2[i])
                ]
                curr_up = 1
                if up == curr_up or up == 0:
                    single_ranked_list.append(group_2)
                    up = 1
                else:
                    multiple_ranked_list.append(single_ranked_list)
                    single_ranked_list = []
                    single_ranked_list.append(group_1)
                    single_ranked_list.append(group_2)
                    up = 1
        multiple_ranked_list.append(single_ranked_list)
    return multiple_ranked_list


def obtain_labels(dataset, idx_1, idx_2, segment_size=25, threshold=0.5, noise=0.0):
    idx_1 = np.array(idx_1)
    idx_2 = np.array(idx_2)
    labels = []
    reward_1 = np.sum(dataset["rewards"][idx_1], axis=1)
    reward_2 = np.sum(dataset["rewards"][idx_2], axis=1)
    labels = np.where(reward_1 < reward_2, 1, 0)
    labels = np.array([[1, 0] if i == 0 else [0, 1] for i in labels]).astype(float)
    gap = segment_size * threshold

    equal_labels = np.where(
        np.abs(reward_1 - reward_2) <= segment_size * threshold, 1, 0
    )
    labels = np.array(
        [labels[i] if equal_labels[i] == 0 else [0.5, 0.5] for i in range(len(labels))]
    )
    if noise != 0.0:
        p = noise
        for i in range(len(labels)):
            if labels[i][0] == 1:
                if random.random() < p:
                    if random.random() < 0.5:
                        labels[i][0] = 0
                        labels[i][1] = 1
                    else:
                        labels[i][0] = 0.5
                        labels[i][1] = 0.5
            elif labels[i][1] == 1:
                if random.random() < p:
                    if random.random() < 0.5:
                        labels[i][0] = 1
                        labels[i][1] = 0
                    else:
                        labels[i][0] = 0.5
                        labels[i][1] = 0.5
            else:
                if random.random() < p:
                    if random.random() < 0.5:
                        labels[i][0] = 0
                        labels[i][1] = 1
                    else:
                        labels[i][0] = 1
                        labels[i][1] = 0
    return labels
