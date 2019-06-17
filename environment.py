import copy

import numpy as np

import gym
from mxnet import nd

class Environment(object):
    """
    Environment wrapper to handle methods such as evaluate, get_trajectory_sync,
    etc.

    Slightly specific to CartPole-v0 environment (being that the solved method
    averages and compares against a threshold).
    """
    def __init__(self, env_id, gamma, lamda_, timesteps, solved_thresh=199.5):
        self._env = gym.make(env_id)
        self._gamma = gamma
        self._lambda = lamda_
        self._timesteps = timesteps
        self._convert_state = lambda state: nd.array([state])
        self._policy = None
        self._gen = None
        self._solved_thresh = solved_thresh

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
        # TODO(Conrad): Might need to seed MxNet and numpy
        self._env.seed(seed)

    def start(self, policy):
        """
        Must call start primarily to pass the policy to the environment. But
        this methods also controls starting the generator for
        get_trajectory_sync.
        """
        self._policy = policy
        self._gen = self._generator()

    def solved(self):
        mean_return = np.mean([self.evaluate(False)[0] for _ in range(10)])
        print('Mean return over 10 episodes: {} ({} to pass)'.format(
            mean_return, self._solved_thresh))
        return mean_return > self._solved_thresh

    def evaluate(self, stochastic=False):
        """
        Stochastic parameter determines if the agent acts based on highest
        perceived reward, or if it samples based on values.
        """
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

    def display(self):
        """
        This is called after training to show off the model

        TODO(Conrad): Abstract out the loop which is also used in evaluate
        """
        ob = self._convert_state(self._env.reset())
        done = False
        while not done:
            ac, _ = self._act(ob, stochastic=False)
            ob, rew, done, _ = self._env.step(ac)
            ob = self._convert_state(ob)
            self._env.render()
        self._env.close()

    def close(self):
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
        """
        Yields a length `T` trajectory

        (Straight from the openAI code, essentially)
        """
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
