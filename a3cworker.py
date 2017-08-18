import os
from collections import OrderedDict
from collections import defaultdict
from collections import deque
from typing import Iterable
import math
import numpy as np

import gym
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
from torch import autograd
from torch import multiprocessing

import vis_graph

def env_gen(env):
    obs = torch.FloatTensor(env.reset())
    action = yield (obs, None, False, {})
    while True:
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
        obs = torch.FloatTensor(obs)
        action = yield (obs, reward, done, info)

def run_env(conn, env):
    obsTensor = torch.FloatTensor(env.reset())
    obsTensor.share_memory_()
    gen = env_gen(env)
    while True:
        obs, reward, done, info = next(gen)
        obsTensor.copy_(obs)
        conn.send((obsTensor, reward, done, info))
        action = conn.recv()
        gen.send(action)

def env_gen_multiprocessing(env):
    conn, runner_conn = multiprocessing.Pipe()
    env_runner = multiprocessing.Process(target=run_env, args=(runner_conn, env))
    env_runner.start()
    while True:
        action = yield conn.recv()
        conn.send(action)


class A3CWorker:
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
                 environment: gym.Env,
                 model: nn.Module,
                 worker_id: int = 0,
                 optimizer: optim.Optimizer=None,
                 min_update_period: int=15,
                 max_update_period: int=30,
                 gamma: float=0.99,
                 log_dir: str=None,
                 checkpoint_dir: str=None,
                 terminate: multiprocessing.Value=None,
                 global_steps: multiprocessing.Value=None,
                 backprop_depth=5):
        self.environment = environment
        self.model_state0 = None
        self.model_state1 = None
        self.min_update_period = min_update_period
        self.max_update_period = max_update_period
        self.gamma = gamma
        self.log_dir = log_dir
        if checkpoint_dir is None:
            self.checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        else:
            self.checkpoint_dir = checkpoint_dir
        self.backprop_depth = backprop_depth
        self.global_model = model
        self.model = model  # model.clone()
        self.discounted_gradients = None
        self.features = deque()
        self.observation = None
        self.prev_value_estimate = None
        self.value_estimate = None
        self.reward = None
        self.log_likelihood = None
        self.entropy = None
        self.local_steps = 0
        self.episode = 0
        self.global_steps = global_steps
        self.terminate = terminate
        self.worker_id = worker_id
        self.info = None
        self.episode_done = False
        self.optimizer = optimizer
        self.worker_id = worker_id
        self.summary_writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'train_{}'.format(worker_id)))
        # self.env_gen = env_gen(self.environment)

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

    def _reset(self):
        self.model_state0 = self.model.get_initial_state()
        self.model_state1 = self.model.get_initial_state()
        self.features.clear()
        self.discounted_gradients = None

    def step(self):
        r"""Advance the simulation by one action. Update the weights if necessary"""
        if self.observation is None:
            obs = self.environment.reset()
            self.observation = self._float_var(obs)
            self._reset()
        # if reward_val is not None:
        #     self.rewards_list.append(reward_val)
        if len(self.features) > 0:
            self._update_weights()
            self._update_gradients()

        self._output_summary()

        if self.episode_done:
            self._reset()
            self.episode += 1

        # if self.update_period <= 0 or self.episode_done:
        #     self._backward_step()
        #     self._update_weights()
        #     self._output_summary()
        #     self.global_steps.value += len(self.log_likelihood_list)
        #     if self.episode_done:
        #         self.model_state = None
        #         self.episode += 1
        #     self._reset()

        self._act()
        self.local_steps += 1
        self.global_steps.value += 1

    def _update_weights(self):
        if self.discounted_gradients is None:
            self.discounted_gradients = [param.data.new(param.data.size()) for param in self.model.parameters()]
        else:
            v1 = self.value_estimate.data.cpu().numpy()[0]
            v0 = self.prev_value_estimate.data.cpu().numpy()[0]
            s = self.reward + self.gamma * v1 - v0
            for (accum, param) in zip(self.discounted_gradients, self.model.parameters()):
                torch.mul(accum, s, out=param.grad.data)
            (0.01 * self.entropy).backward(retain_graph=True)
            utils.clip_grad_norm(self.model.parameters(), 40)
            self.optimizer.step()

    def _update_gradients(self):
        self.model.zero_grad()
        unscaled_loss = self.log_likelihood - 0.5 * self.value_estimate
        unscaled_loss.backward(retain_graph=True)
        for (accum, param) in zip(self.discounted_gradients, self.model.parameters()):
            accum.mul_(self.gamma)
            accum.add_(param.grad.data)
            pass
        if len(self.features) >= self.backprop_depth:
            feature0 = self.features.popleft()
            self.model_state0 = self.model.forward_state([feature0], self.model_state0).detach()
        self.model_state1 = self.model.forward_state(self.features, self.model_state0)

    def _act(self):
        r"""Sample from the current policy and perform an action"""
        log_prob, value, feature = self.model(self.observation, self.model_state1)
        self.features.append(feature)
        prob = torch.exp(log_prob)
        self.entropy = (prob * log_prob).sum()

        self.action = prob.data.multinomial(1).cpu().numpy()[0]

        self.prev_value_estimate = self.value_estimate
        self.value_estimate = value
        self.log_likelihood = log_prob[self.action]

        obs, self.reward, self.episode_done, self.info = self.environment.step(self.action)
        if self.episode_done:
            obs = self.environment.reset()
        self.observation = self._float_var(obs)

        # self.model_states.append(state)

    # def _backward_step(self):
    #     r"""Backpropogate the loss from the current batch, computing the gradients"""
    #     loss = self._loss()
    #     self.model.zero_grad()
    #     loss.backward()
    #     utils.clip_grad_norm(self.model.parameters(), 40)
    #
    # def _loss(self) -> autograd.Variable:
    #     r"""Compute the loss function for the current batch"""
    #
    #     self._get_final_value()
    #     policy_loss = self._get_policy_loss()
    #     value_loss = self._get_value_loss()
    #     loss = policy_loss + 0.5 * value_loss - 0.01 * self.entropy
    #
    #     batch_size = len(self.log_likelihood_list)
    #     if self._should_compute_summary():
    #         self.info["model/policy_loss"] = policy_loss.data.cpu().numpy()[0] / batch_size
    #         self.info["model/value_loss"] = value_loss.data.cpu().numpy()[0] / batch_size
    #
    #     return loss
    #
    # def _get_value_loss(self) -> autograd.Variable:
    #     r"""Computes the loss function for the value estimates"""
    #     discounted_rewards = self._discount(self.rewards_var.data.cpu().numpy())
    #     return 0.5 * ((self.value_estimates_var - discounted_rewards) ** 2).sum()
    #
    # def _get_policy_loss(self) -> autograd.Variable:
    #     r"""Computes the loss function for the policy"""
    #     advantage_delta = self.rewards_var[:-1] - self.value_estimates_var[:-1] + self.gamma * self.value_estimates_var[1:]
    #     advantage = self._discount(advantage_delta.data.cpu().numpy())
    #     return -torch.sum(self.log_likelihood_var * advantage)
    #
    # def _get_final_value(self):
    #     r"""Computes the estimated value of the observation in the batch"""
    #     if self.episode_done:
    #         final_value = self._float_var([0.0])
    #     else:
    #         _, final_value, _ = self.model(self.observation, self.model_state)
    #     self.value_estimates_list.append(final_value)
    #     self.rewards_list.append(final_value.data[0])
    #
    #     def cat_vars(vars) -> autograd.Variable:
    #         return torch.cat([v for v in vars])
    #
    #     self.log_likelihood_var = cat_vars(self.log_likelihood_list)  # size: (batch,)
    #     self.value_estimates_var = cat_vars(self.value_estimates_list)  # size: (batch,)
    #     self.rewards_var = self._float_var(self.rewards_list)  # size: (batch,)

    # def _update_weights(self):
    #     r"""Updates the global weights using the current local gradients"""
    #
    #     # self.global_model.load_gradients(self.model)
    #     self.optimizer.step()

    def _should_compute_summary(self):
        r"""Used to control the rate at which statistics are logged

        Returns:
            bool: True if this step should be logged, false otherwise
        """
        return self.local_steps % 20 == 1

    def _output_summary(self):
        r"""Writes out the relevant statistics, viewable in tensorboard"""
        self.info = self.info or {}
        if self._should_compute_summary():
            self.info["model/grad_global_norm"] = self.global_model.grad_norm()
            self.info["model/param_global_norm"] = self.model.param_norm()
            self.info["model/entropy"] = self.entropy.data.cpu().numpy()[0]

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
        if self.model.is_cuda:
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