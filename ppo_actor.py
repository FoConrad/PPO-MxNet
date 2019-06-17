# PPO code inspired by and borrowed from https://github.com/dai-dao/PPO-Gluon
# and https://github.com/openai/baselines/tree/master/baselines/ppo1
import time
from collections import deque

import numpy as np
from mxnet import gluon, nd, autograd

from policies import MLPPolicy

class PPO(object):
    def __init__(self, env, action_dim, batch_size, expr_name, seed=42,
                 layer_sizes=[128], log_rate_iters=10, policy='mlp',
                 **policy_args):
        self.env = env
        self.env.seed(seed)
        self.action_dim = action_dim
        # FIXME(fegin): It is necessary due to the implementation of seq2seq.
        #               Check if we can remove this field by rewriting seq2seq.
        self.batch_size = batch_size
        self._layer_sizes = layer_sizes
        self._policy_args = policy_args
        self._log_rate = log_rate_iters

        self._policy = policy

    def policy_fn(self):
        assert self._policy == 'mlp', 'Other policies not implanted yet'
        return MLPPolicy(action_dim=self.action_dim,
                         layer_sizes=self._layer_sizes)

    def update(self, obs, returns, actions, advantages, cliprange_now,
            entropy_coeff):
        advantages = nd.array(advantages)
        actions = nd.array(actions)
        returns = nd.array(returns)

        with autograd.record():
            _, old_logits = self.oldpi.forward(obs)
            old_logp = self.oldpi.logp(old_logits, actions)

            new_vpred, new_logits = self.pi.forward(obs)
            new_vpred = new_vpred.reshape(new_vpred.shape[:-1])
            new_logp = self.pi.logp(new_logits, actions)

            # Action loss
            ratio = nd.exp(new_logp - old_logp)
            surr1 = ratio * advantages
            surr2 = nd.clip(ratio, 1.0 - cliprange_now, 1.0 + cliprange_now) * advantages
            actor_loss = -nd.mean(nd.minimum(surr1, surr2))

            # Value loss
            vf_loss1 = nd.square(new_vpred - returns)
            vf_loss = nd.mean(vf_loss1)

            # Entropy term
            entrpy = self.pi.entropy(new_logits)
            mean_entrpy = nd.mean(entrpy)
            ent_loss = (-entropy_coeff) * mean_entrpy

            loss = vf_loss + actor_loss + ent_loss

        # Compute gradients and updates
        loss.backward()
        self.trainer.step(1) # Loss are already normalized

        return actor_loss.asscalar(), vf_loss.asscalar(), ent_loss.asscalar()

    def learn(self, *,
              clip_param, entcoeff,  # Clipping parameter epsilon, entropy coeff
              optim_epochs, optim_stepsize, optim_batchsize,  # Optimization hyperparameters
              max_episodes=0,        # Time constraint
              callback=None,         # callback(locals(), globals()) called after iteration end
              adam_epsilon=1e-5,
              schedule='linear',     # Annealing for stepsize parameters (epsilon, LR, entcoeff),
              anneal_entcoeff=True   # Anneal the entropy regularization when True
              ):
        env = self.env
        max_timesteps = max_episodes*100  # TODO get episode length instead of 100

        # Setup losses etc. (TF ops)
        # ----------------------------------------
        self.pi = self.policy_fn()
        self.oldpi = self.policy_fn()

        self.trainer = gluon.Trainer(self.pi.collect_params(), 'adam',
                {'learning_rate': optim_stepsize, 'epsilon': adam_epsilon})

        if hasattr(env, 'start'):
            env.start(self.pi)

        # Prepare for rollouts
        # ----------------------------------------
        episodes_so_far = 0
        timesteps_so_far = 0
        iters_so_far = 0
        tstart = time.time()
        lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
        rewbuffer = deque(maxlen=100)
        queries = []

        while True:
            if max_episodes and episodes_so_far >= max_episodes:
                break

            if schedule == 'constant':
                cur_lrmult = 1.0
            elif schedule == 'linear':
                cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
            else:
                raise NotImplementedError

            cur_cliprange = clip_param * cur_lrmult
            cur_entcoeff = entcoeff * cur_lrmult if anneal_entcoeff else entcoeff
            self.trainer.set_learning_rate(optim_stepsize * cur_lrmult)

            # Do actual training and parameter updates
            # ----------------------------------------
            # `seg` contains `timesteps_per_update` steps of experience
            seg = env.get_trajectory_sync()
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], \
                    seg["tdlamret"]
            # Normalize advantages
            atarg = (atarg - atarg.mean()) / atarg.std()

            self.oldpi.assign_parameters(self.pi.collect_params())

            nbatch = len(ob)
            inds = np.arange(nbatch)
            minibatch_lossvals = []
            for _ in range(optim_epochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, optim_batchsize):
                    end = start + optim_batchsize
                    batch_inds = inds[start : end]
                    slices = (arr[batch_inds] for arr in (ob, tdlamret, ac,
                        atarg))
                    policy_loss, value_loss, entropy = self.update(*slices,
                            cur_cliprange, cur_entcoeff)
                    minibatch_lossvals.append([policy_loss, value_loss, entropy])

            lenrewlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
            lens, rews = lenrewlocal
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1
            mbl = np.array(minibatch_lossvals)
            pg_mean = np.mean(mbl[:, 0])
            vf_mean = np.mean(mbl[:, 1])
            entropy_mean = np.mean(mbl[:, 2])
            meanlosses = [pg_mean, vf_mean, entropy_mean]
            loss_names = ['pol_surr', 'vf_loss', 'pol_entpen']

            if callback is not None:
                callback(locals())

            if iters_so_far % self._log_rate == 0 and self.env.solved():
                break

        try:
            # Try rendering
            env.display()
        except:
            print('Some rendering error')
        self.env.close()
