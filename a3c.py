import os
from collections import OrderedDict
from collections import defaultdict
from typing import Iterable

import gym
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
from torch import autograd
from torch import multiprocessing


class A3C:
    r"""Implementation of the A3C reinforcement learning algorithm for use with the OpenAI Gym/Universe.

    The algorithm is from the following paper:

    Mnih, et al. "Asynchronous methods for deep reinforcement learning."
    International Conference on Machine Learning. 2016.

    Args:
        env: environment
        model: model to train
        optimizer: optimizer to use for training. Defaults to RMSProp.
        update_period: number of actions to taken before updating the weights.
        log_dir: path to the directory storing logging info
        terminate: a shared value, containing a bool which, when true, will signal the worker to exit
        global_steps: a shared value, containing the total number of steps taken

    """
    def __init__(self,
                 environments: Iterable[gym.Env],
                 model: nn.Module,
                 optimizer: optim.Optimizer=None,
                 min_update_period: int=15,
                 max_update_period: int=30,
                 gamma: float=0.99,
                 log_dir: str=None,
                 checkpoint_dir: str=None,
                 terminate: multiprocessing.Value=None,
                 global_steps: multiprocessing.Value=None):
        self.environments = list(environments)
        self.environment_id = 0
        self.states = [None] * len(self.environments)
        self.observations = [None] * len(self.environments)
        self.min_update_period = min_update_period
        self.max_update_period = max_update_period
        self.gamma = gamma
        self.is_cuda = model.is_cuda
        self.log_dir = log_dir
        if checkpoint_dir is None:
            self.checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        else:
            self.checkpoint_dir = checkpoint_dir
        self.render = False
        self.global_model = model
        self.value_estimates = []
        self.actions = []
        self.rewards = []
        self.local_steps = 0
        self.episode = 0
        self.global_steps = global_steps
        self.terminate = terminate
        self.worker_id = None
        self.info = None
        self.episode_done = False
        self.optimizer = optimizer
        self.log_likelihood = []
        self.entropy = 0

    def weights_path(self):
        r"""Path to file where model weights are saved/loaded

        """
        return os.path.join(self.log_dir, "weights")

    def state_dict(self, destination=None):
        """Returns the model weights and optimizer state as a :class:`dict`.

        """
        if destination is None:
            destination = OrderedDict()
        destination['model'] = self.global_model.state_dict()
        destination['optimizer'] = self.optimizer.state_dict()
        destination['global_steps'] = self.global_steps.value
        return destination

    def load_state_dict(self, state_dict):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. The keys of :attr:`state_dict` must
        exactly match the keys returned by this module's :func:`state_dict()`
        function.

        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.

        """
        if set(state_dict.keys()) != set(self.state_dict().keys()):
            raise KeyError("State dict does not contain the correct set of keys. Expected {}, received {}".format(
                set(self.state_dict().keys()),
                set(state_dict.keys())))

        self.global_model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.optimizer.state = defaultdict(dict)  # bug workaround in Optimizer.load_state_dict
        self.global_steps.value = state_dict['global_steps']

    def save_checkpoint(self):
        r"""Save a checkpoint file, to recover current training state"""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.isdir(self.checkpoint_dir):
            raise IOError("Checkpoint directory path is not a directory")
        path = os.path.join(self.checkpoint_dir, '{}.ckpt'.format(self.global_steps.value))
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, checkpoint_id=None):
        r"""Load the model weights from a file"""
        if checkpoint_id is None:
            if not os.path.exists(self.checkpoint_dir):
                return False
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.ckpt')]
            if len(checkpoint_files) == 0:
                return False
            checkpoint_ids = [int(os.path.splitext(f)[0]) for f in checkpoint_files]
            checkpoint_id = max(checkpoint_ids)
        checkpoint_path = os.path.join(self.checkpoint_dir, '{}.ckpt'.format(checkpoint_id))
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict)
        return True

    def run(self, worker_id: int = 0, render: bool =None, optimizer=None):
        r""" Entry point for worker.

        Args:
            worker_id: a unique identifier for this worker. Should be in `range(num_workers)`
            render: overrides the render field for this worker.

        """
        if callable(self.optimizer):
            self.optimizer = self.optimizer()
        self.worker_id = worker_id
        if render is not None:
            self.render = render
        if optimizer is not None:
            self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = optim.RMSprop(self.global_model.parameters(), lr=1e-3)

        self.model = self.global_model.clone()
        self.summary_writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'train_{}'.format(worker_id)))

        while not self.terminate.value:
            self._step()

    def _step(self):
        r"""Runs the simulation long enough to update the model weights once"""
        self.log_likelihood = []
        self.entropy = 0
        self.rewards = []
        self.value_estimates = []
        self.model.load_parameters(self.global_model)
        update_period = np.random.randint(self.min_update_period, self.max_update_period)
        for _ in range(update_period):
            if self.render:
                self._environment().render()
            self._act()
            if self.episode_done:
                break
        self._backward_step()
        self._update_weights()
        self._output_summary()
        self.global_steps.value += len(self.log_likelihood)
        self.local_steps += 1
        if self.episode_done:
            self.episode += 1
            self._set_state(None)
        else:
            self._state().detach_()
        self._next_environment()

    def _act(self):
        r"""Sample from the current policy and perform an action"""
        log_prob, value, state = self.model(self._observation(), self._state())
        self._set_state(state)
        prob = torch.exp(log_prob)
        self.entropy -= (prob * log_prob).sum()

        action = prob.data.multinomial(1).cpu().numpy()[0]
        observation, reward_val, done, info = self._environment().step(action)
        self._set_observation(observation)

        self.rewards.append(reward_val)
        self.value_estimates.append(value)
        self.log_likelihood.append(log_prob[action])

        if done:
            self._set_observation(None)

        self.episode_done = done
        self.info = info or {}

    def _backward_step(self):
        r"""Backpropogate the loss from the current batch, computing the gradients"""
        loss = self._loss()
        self.model.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.model.parameters(), 40)

    def _loss(self):
        r"""Compute the loss function for the current batch"""

        self._get_final_value()
        policy_loss = self._get_policy_loss()
        value_loss = self._get_value_loss()
        loss = policy_loss + 0.5 * value_loss - 0.01 * self.entropy

        batch_size = len(self.log_likelihood)
        if self._should_compute_summary():
            self.info["model/policy_loss"] = policy_loss.data.cpu().numpy()[0] / batch_size
            self.info["model/value_loss"] = value_loss.data.cpu().numpy()[0] / batch_size

        return loss

    def _get_value_loss(self):
        r"""Computes the loss function for the value estimates"""
        discounted_rewards = self._discount(self.rewards.data.cpu().numpy())
        return 0.5 * ((self.value_estimates - discounted_rewards) ** 2).sum()

    def _get_policy_loss(self):
        r"""Computes the loss function for the policy"""
        advantage_delta = self.rewards[:-1] - self.value_estimates[:-1] + self.gamma * self.value_estimates[1:]
        advantage = self._discount(advantage_delta.data.cpu().numpy())
        return -torch.sum(self.log_likelihood * advantage)

    def _get_final_value(self):
        r"""Computes the estimated value of the observation in the batch"""
        if self.episode_done:
            final_value = self._float_var([0.0])
        else:
            _, final_value, _ = self.model(self._observation(), self._state())
        self.value_estimates.append(final_value)
        self.rewards.append(final_value.data[0])

        def cat_vars(vars):
            return torch.cat([v.unsqueeze(0) for v in vars])

        self.log_likelihood = cat_vars(self.log_likelihood)  # size: (batch,)
        self.value_estimates = cat_vars(self.value_estimates)  # size: (batch,)
        self.rewards = self._float_var(self.rewards)  # size: (batch,)

    def _update_weights(self):
        r"""Updates the global weights using the current local gradients"""

        self.global_model.load_gradients(self.model)
        self.optimizer.step()

    def _should_compute_summary(self):
        r"""Used to control the rate at which statistics are logged

        Returns:
            bool: True if this step should be logged, false otherwise
        """
        return self.local_steps % 10 == 0

    def _output_summary(self):
        r"""Writes out the relevant statistics, viewable in tensorboard"""
        if self._should_compute_summary():
            self.info["model/grad_global_norm"] = self.global_model.grad_norm()
            self.info["model/param_global_norm"] = self.model.param_norm()

        summary = tf.Summary()
        for k, v in self.info.items():
            summary.value.add(tag=k, simple_value=float(v))
        self.summary_writer.add_summary(summary, self.global_steps.value)
        self.summary_writer.flush()

    def _float_var(self, x):
        r"""Converts an `array_like` object to a `torch.Variable`

        Args:
            x (array_like): an object which can be converted to a numpy array of `float`s

        Returns:
            torch.Varible: A variable containing the data `x`, as a `torch.FloatTensor`.
             This variable will be on the GPU, if `self.is_cuda` is True.
        """
        x = torch.FloatTensor(np.array(x))
        if self.is_cuda:
            x = x.cuda()
        return autograd.Variable(x)

    def _long_var(self, x):
        r"""Converts an `array_like` object to a `torch.Variable`

        Args:
            x (array_like): an object which can be converted to a numpy array of `long`s

        Returns:
            torch.Varible: A variable containing the data `x`, as a `torch.LongTensor`.
             This variable will be on the GPU, if `self.is_cuda` is True.
        """
        x = torch.LongTensor(np.array(x))
        if self.is_cuda:
            x = x.cuda()
        return autograd.Variable(x)

    def _discount(self, rewards: np.ndarray):
        r"""Apply reward discounting.

        Args:
            rewards: the raw rewards

        Returns:
            the discounted rewards
        """
        d = np.zeros_like(rewards)
        d[-1] = rewards[-1]
        for i in range(len(rewards)-2, -1, -1):
            d[i] = rewards[i] + d[i + 1] * self.gamma
        return self._float_var(d)

    def _environment(self):
        r"""Get the current environment"""
        return self.environments[self.environment_id]

    def _state(self):
        r"""Get the model state for the current environment"""
        if self.states[self.environment_id] is None:
            self.states[self.environment_id] = self.model.get_initial_state()
        return self.states[self.environment_id]

    def _set_state(self, state):
        r"""Set the model state for the current environment"""
        self.states[self.environment_id] = state

    def _next_environment(self):
        r"""Advance to the next environment"""
        self.environment_id += 1
        self.environment_id %= len(self.environments)

    def _observation(self):
        r"""Get the observation for the current environment"""
        if self.observations[self.environment_id] is None:
            self.observations[self.environment_id] = self._float_var(self._environment().reset())
        return self.observations[self.environment_id]

    def _set_observation(self, observation):
        r"""Set the observation for the current environment"""
        if observation is not None:
            observation = self._float_var(observation)
        self.observations[self.environment_id] = observation