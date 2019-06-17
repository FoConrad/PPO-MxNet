#!/usr/bin/env python
import argparse
import logging
import math
import os
import time
import copy
from collections import deque

import gym
import numpy as np
from mxnet import nd

import utils
from ppo_actor import PPO


class Environment(object):
    def __init__(self, env_id, gamma, lamda_, timesteps):
        self._env = gym.make(env_id)
        self._gamma = gamma
        self._lambda = lamda_
        self._timesteps = timesteps
        self._convert_state = lambda state: nd.array([state])
        self._policy = None
        self._gen = None

    @property
    def started(self):
        return self._policy is not None

    @property
    def action_dim(self):
        # Assumes discrete action space
        assert isinstance(self._env.action_space,
                          gym.spaces.discrete.Discrete)

        return self._env.action_space.n

    def seed(self, seed):
        # TODO(Conrad): Is there a need to seed MxNet or np?
        self._env.seed(seed)

    def start(self, policy):
        self._policy = policy
        self._gen = self._generator()

    def evaluate(self, *args):
        ob = self._converter.state_from_gym(self._env.reset())
        done = False
        while not done:
            ac = self._act(ob, explore=-1)
            ob, cost, done, _ = self._env.step(ac)
            ob = self._converter.state_from_gym(ob)
        return self._env.raw_signal(), self._env.cost_model

    def evaluate(self):
        def _evaluate(stochastic):
            ob = self._convert_state(self._env.reset())
            done = False
            actions = []
            sum_rew = 0
            while not done:
                ac, _ = self._act(ob, stochastic=stochastic)
                actions.append(ac)
                ob, rew, done, _ = self._env.step(ac)
                ob = self._convert_state(ob)
                sum_rew += rew
            self.ob = self._convert_state(self._env.reset())
            self.new = True
            return sum_rew, actions

        return _evaluate(stochastic=False), _evaluate(stochastic=True)

    def finish(self):
        ob = self._convert_state(self._env.reset())
        done = False
        while not done:
            ac, _ = self._act(ob, stochastic=False)
            ob, rew, done, _ = self._env.step(ac)
            ob = self._convert_state(ob)
            self._env.render()
        self._env.close()

    def get_trajectory_sync(self):
        return next(self._gen)

    def _act(self, observation, stochastic=True):
        # If there are multiple workers, they will almost for sure call the
        # policy at the same time. This causes an issue in MXNet. Here should
        # be the only place where this may happen
        #with self._lock:
        action, value = self._policy.act(stochastic, observation)
        # Check if batched act or not
        if np.prod(np.array(action.shape)) == 1:
            return action.asscalar(), value[0].asscalar()
        else:
            return action.asnumpy(), value[0].asscalar()

    @staticmethod
    def _add_vtarg_and_adv(seg, gamma, lam):
        """Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)"""
        new = np.append(seg["new"], 0)
        vpred = np.append(seg["vpred"], seg["nextvpred"])
        T = len(seg["rew"])
        seg["adv"] = gaelam = np.empty(T, 'float32')
        rew = seg["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[t + 1]
            delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        seg["tdlamret"] = seg["adv"] + seg["vpred"]
        del seg["nextvpred"]


    def _generator(self):
        """Yields a length `T` trajectory"""
        # Initial setup
        ac = self._env.action_space.sample()  # not used, just so we have the datatype
        self.new = True  # marks if we're on first timestep of an episode
        self.ob = self._convert_state(self._env.reset()) 
        T = self._timesteps

        cur_ep_ret = 0  # return in current episode
        cur_ep_len = 0  # len of current episode
        ep_rets = []  # returns of completed episodes in this segment
        ep_lens = []  # lengths of ...

        # Initialize history arrays
        #obs = np.array([None for _ in range(T)])
        obs = nd.empty((T,) + self._env.observation_space.shape)
        rews = np.zeros(T, 'float32')
        vpreds = np.zeros(T, 'float32')
        news = np.zeros(T, 'int32')
        acs = np.array([ac for _ in range(T)])
        prevacs = acs.copy()

        t = 0
        while True:
            ob = self.ob  # Use `self.` since `_evaluate` may have reset the env
            new = self.new
            prevac = ac
            ac, vpred = self._act(ob)
            # NOTE(openAI) Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct terminal value
            if t > 0 and t % T == 0:
                seg = {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                       "ac": acs, "nextvpred": vpred * (1 - new),
                       "ep_rets": np.array(copy.deepcopy(ep_rets)),
                       "ep_lens": np.array(copy.deepcopy(ep_lens))}
                self._add_vtarg_and_adv(seg, self._gamma, self._lambda)
                yield seg
                # NOTE: Do a deepcopy if the values formerly in these arrays are used later.
                ep_rets = []
                ep_lens = []
            i = t % T

            obs[i] = ob[0]
            vpreds[i] = vpred
            news[i] = new
            acs[i] = ac
            prevacs[i] = prevac

            ob, rew, new, _ = self._env.step(ac)
            ob = self._convert_state(ob)
            rews[i] = rew

            cur_ep_ret += rew
            cur_ep_len += 1
            if new:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                ob = self._convert_state(self._env.reset())
            self.new = new
            self.ob = ob
            t += 1


def iteration_callback(locals_):
    """Log, query model and visualize."""
    print("********** Iteration {:,} ({:,} episodes) ************".format(
                locals_['iters_so_far'], locals_['episodes_so_far']))
    self = locals_['self']
    if locals_['iters_so_far'] % self._log['rate'] == 0:
        # Query
        (cost, actions), (s_cost, s_actions) = self.env.evaluate()
        eps = locals_['episodes_so_far']
        elapsed_time = (time.time() - locals_['tstart'])
        locals_['queries'].append((cost, eps, elapsed_time))

        # Print logging
        print("=== Stochastic === \n" +
              "episodes: {:,}, cost: {:,}".format(eps, s_cost))
              #"actions: {}".format(s_actions))
        print("=== Greedy ===\n" +
              "episodes: {:,}, cost: {:,}\n".format(eps, cost))
              #"actions: {}".format(actions))
        print("Elapsed time: {}".format(elapsed_time))
        if cost == 200:
            raise ValueError('Reached peak performance')


def setup_arguments():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Environment parameters
    parser.add_argument('--env-type', default='cartpole', choices=('cartpole',))
    parser.add_argument('--policy', default='mlp', choices=('mlp',))

    # Physical environment
    parser.add_argument('--with-gpu', action='store_true', help='train model on GPU')

    # Misc
    parser.add_argument('--seed', help='RNG seed', type=int, default=42)
    parser.add_argument('--noise', type=float, default=0.0, help='perturb reward with noise')

    # Logging
    parser.add_argument('--base-log-dir', default=None,
                        help='Enables logging when not None')
    parser.add_argument('--log-rate', type=int, default=10,
                        help='log (query model, visualize) every log-rate iterations')
    parser.add_argument('--expr-name', default='rltofu', help='prefix for log directory')

    # Network hyperparameters
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='number of hidden units in policy and value network')
    parser.add_argument('--n-hidden-layers', type=int, default=1,
                        help='number of layers for networks')

    # PPO hyperparameters
    parser.add_argument('--max-episodes', type=int, default=5000,
                        help='terminate after this many episodes')
    parser.add_argument('--entcoeff', help='Entropy reg. coefficient', type=float, default=0.2)
    parser.add_argument('--timesteps-per-update', type=int, default=256,
                        help='number of steps to take before PPO training')
    parser.add_argument('--clip-param', help='Gradient clipping', type=float, default=0.2)
    parser.add_argument('--optim-epochs', type=int, default=4,
                        help='number of times to pass over timesteps of data')
    parser.add_argument('--optim-stepsize', type=float, default=1e-3,
                        help='learning rate for networks')
    parser.add_argument('--optim-batchsize', type=int, default=64,
                        help='size of mini-batch for PPO')
    parser.add_argument('--gamma', help='discount factor 1', type=float, default=0.99)
    parser.add_argument('--lam', help='discount factor 2', type=float, default=0.95)
    parser.add_argument('--schedule', choices=['linear', 'constant'], default='linear',
                        help='decay schedule for learning rate and entropy')

    args = parser.parse_args()

    return args


def main():
    args = setup_arguments()
    cfg = utils.setup(args)

    # --- Env setup
    env = Environment('CartPole-v0', args.gamma, args.lam, args.timesteps_per_update)
    action_dim = env.action_dim


    # ---
    ppo = PPO(env,
              batch_size=args.optim_batchsize,
              action_dim=action_dim,
              log_dir=cfg.get('log_directory', None),
              expr_name=args.expr_name,
              seed=args.seed,
              layer_sizes=[args.hidden_dim]*args.n_hidden_layers,
              log_rate_iters=args.log_rate,
              policy=args.policy)

    ppo.learn(max_episodes=args.max_episodes,
              entcoeff=args.entcoeff,
              clip_param=args.clip_param,
              optim_epochs=args.optim_epochs,
              optim_stepsize=args.optim_stepsize,
              optim_batchsize=args.optim_batchsize,
              schedule=args.schedule,
              callback=iteration_callback)


if __name__ == '__main__':
    main()
