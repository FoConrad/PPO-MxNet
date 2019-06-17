#!/usr/bin/env python
import argparse
import time

from environment import Environment
from ppo_actor import PPO


def iteration_callback(locals_):
    """Log, query model and visualize."""
    print("********** Iteration {:,} ({:,} episodes) ************".format(
                locals_['iters_so_far'], locals_['episodes_so_far']))
    self = locals_['self']
    if locals_['iters_so_far'] % self._log_rate == 0:
        # Query
        cost, actions = self.env.evaluate(False)
        s_cost, s_actions = self.env.evaluate(True)

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

    # --- Env setup
    env = Environment('CartPole-v0', args.gamma, args.lam, args.timesteps_per_update)
    action_dim = env.action_dim

    # ---
    ppo = PPO(env,
              batch_size=args.optim_batchsize,
              action_dim=action_dim,
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
