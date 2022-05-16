from cmath import inf
from collections import namedtuple
from dis import dis
from multiprocessing.sharedctypes import Value
import os
import sys
from tkinter.messagebox import NO
from xmlrpc.client import Boolean
from matplotlib.pyplot import axis
import numpy as np
import pickle

from torch import absolute
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')


# LEARNING

Experiment = namedtuple('Experiment', 'values parameters')
Values = namedtuple('Values', 'rewards succeded')


def load_experiment(
    path_data: str,
    path_name: str
) -> tuple:
    rewards = np.load(path_data + '/'+ path_name + '/log/rewards.npy')
    succeded = np.load(path_data + '/' + path_name + '/log/succeded.npy')
    with open(path_data + '/' + path_name + '/log/parameters.pkl', 'rb') as f:
        parameters = pickle.load(f)
    return Experiment(Values(rewards, succeded), parameters)


def load_multiple_experiments(
    paths_data: list,
    paths_name: list
) -> list:
    assert len(paths_data) == len(paths_name), 'ERROR: paths dimension missmatch.'
    logs = []
    for i in range(len(paths_name)):
        logs.append(
            load_experiment(paths_data[i], paths_name[i]))
    return logs


def combine_experiment_values(
    experiments: list
) -> tuple:
    experiments_count = len(experiments)
    if experiments_count < 2:
        return Values(experiments[0].values.rewards, 
                      experiments[0].values.succeded)
    # get smalest amount of epochs
    min_epochs = np.inf
    for i in range(experiments_count):
        if experiments[i].values.rewards.shape[0] < min_epochs:
            min_epochs = experiments[i].values.rewards.shape[0]
    # concatenate arrays with minimal shape
    rewards = experiments[0].values.rewards.copy()[:min_epochs]
    succeded = experiments[0].values.succeded.copy()[:min_epochs]
    for i in range(1, experiments_count):
        rewards = np.concatenate(
            (rewards, experiments[i].values.rewards[:min_epochs]), 
            axis=1)
        succeded = np.concatenate(
            (succeded, experiments[i].values.succeded[:min_epochs]), 
            axis=1)        
    return Values(rewards, succeded)


def average_experiment_values(
    experiments: list
) -> tuple:
    experiments_count = len(experiments)
    if experiments_count < 2:
        return Values(experiments[0].values.rewards, 
                      experiments[0].values.succeded)
    # get smalest amount of epochs
    min_epochs = np.inf
    for i in range(experiments_count):
        if experiments[i].values.rewards.shape[0] < min_epochs:
            min_epochs = experiments[i].values.rewards.shape[0]
    # concatenate arrays with minimal shape
    rewards = np.reshape(
        np.mean(
            experiments[0].values.rewards.copy()[:min_epochs], 
            axis=1), 
        (-1, 1))
    succeded = np.reshape(
        np.mean(
            experiments[0].values.succeded.copy()[:min_epochs].astype(int), 
            axis=1), 
        (-1, 1))
    for i in range(1, experiments_count):
        rewards = np.concatenate(
            (rewards, 
            np.reshape(
                np.mean(
                    experiments[i].values.rewards[:min_epochs], 
                    axis=1), 
                (-1, 1))), 
            axis=1)
        succeded = np.concatenate(
            (succeded, 
            np.reshape(
                np.mean(
                    experiments[i].values.succeded[:min_epochs].astype(int), 
                    axis=1), 
                (-1, 1))), 
            axis=1)        
    return Values(rewards, succeded)

def mean_values(
    values: tuple
) -> tuple:
    mean_rewards = np.mean(values.rewards, axis=1)
    mean_succeded = np.mean(values.succeded, axis=1)
    return Values(mean_rewards, mean_succeded)


def std_values(
    values: tuple
) -> tuple:
    std_rewards = np.std(values.rewards, axis=1)
    std_succeded = np.std(values.succeded, axis=1)
    return Values(std_rewards, std_succeded)


def min_values(
    values: tuple
) -> tuple:
    min_rewards = np.min(values.rewards, axis=1)
    min_succeded = np.min(values.succeded, axis=1)
    return Values(min_rewards, min_succeded)


def max_values(
    values: tuple
) -> tuple:
    max_rewards = np.max(values.rewards, axis=1)
    max_succeded = np.max(values.succeded, axis=1)
    return Values(max_rewards, max_succeded)


def find_absolute_episode(
    values: np.ndarray,
    absolute: int
) -> int:
    if absolute >= 0:
        return np.argwhere(values >= absolute)[0]
    else:
        return np.argwhere(values < absolute)[0]
    return None


# EVALUATION

def get_traveled_distance(
    data: dict
    ) -> float:
    distance_total = 0
    for i in range(len(data) - 1):
        xy_current = np.concatenate((data[i]['x'], data[i]['y']))
        xy_future = np.concatenate((data[i+1]['x'], data[i+1]['y']))
        distance_total += np.linalg.norm(xy_future - xy_current)
    return distance_total

def get_evaluation(
    path_data: str,
    identifier: str,
    individual_count: int,
    world_count: int,
    experiment_count: int,
    single: bool=False,
    ) -> tuple:
    if single:
        individual_start = individual_count
    else:
        individual_start = 1
    succeded_world = np.array([0, 0, 0, 0])
    steps = []
    distances = []
    distance_optimal = np.linalg.norm(np.array([5, 5])) - 0.5
    for cid in range(individual_start, individual_count+1):
        for wid in range(0, world_count):
            path_name = 'EVAL-{}-{}-{}'.format(wid, identifier, cid)
            for id in range(0, experiment_count):
                # load collected data
                finished = np.load(path_data + '/' + path_name + '/log/finished-{}.npy'.format(id))
                succeded = np.load(path_data + '/' + path_name + '/log/succeded-{}.npy'.format(id))
                with open(path_data + '/' + path_name + '/log/data-{}.pkl'.format(id), 'rb') as f:
                    data = pickle.load(f)
                # computing results
                success = np.sum(succeded)
                succeded_world[wid] += success
                if success == 1:
                    steps.append(len(data))
                    dist = get_traveled_distance(data)
                    distances.append(dist)
    # to arrays
    steps = np.array(steps)
    distances = np.array(distances)
    return succeded_world, steps, distances
