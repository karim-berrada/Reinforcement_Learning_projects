from random import random
from math import log

import numpy as np

from multi_armed_bandit.formalization import ArmBernoulli, ArmBeta, ArmGamma, ArmExp


def kl(x, y):

    return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))


class Game:

    def __init__(self, K, perso=False, perso_means=[], expo=False):
        """
        :param K: number of arms (integer)
        """
        self.K = K
        self.expo = expo
        if perso:
            self.means = perso_means
        else:
            self.means = [random() for i in range(K)]

        if expo:
            means = [ArmGamma(0.5, 1).sample() for i in range(K)]
            self.means = means
            self.MAB = [ArmExp(means[i]) for i in range(K)]
        else:
            self.MAB = [ArmBernoulli(self.means[i]) for i in range(K)]

    def initialize(self):

        rewards = [arm.sample() for arm in self.MAB]
        number_draws = np.array([1] * self.K)

        return number_draws, rewards

    def UCB1(self, T, naive=False):

        """
        The UCB1 algorithm starts with an initialization phase that draws each arm once, and then for each t more than K,
        it chooses the arm to draw at t+1 according to a optimal function to maximize
        :param T:
        :param MAB:
        :return:
        """
        # Initialization
        number_draws, rewards = self.initialize()
        rew = []
        draw = []

        for t in range(T):
            print("len = {}".format(t))
            if naive:
                opt_func = rewards / number_draws
            else:
                opt_func = rewards / number_draws + np.sqrt(np.log(t + 1) / (2. * number_draws))
            print("optimization function from which we get the argmax: {}".format(opt_func))

            # Get the argmax from the optimization function
            next_action = np.argmax(opt_func)
            print("Next Arm to draw: {}".format(next_action + 1))

            next_arm = self.MAB[next_action]
            r = next_arm.sample()
            print("Reward of the next arm drawn: {}".format(r))

            # Updating the N(t) and S(t)
            number_draws[next_action] += 1
            print("N vector updated: {}".format(number_draws))

            rewards[next_action] += r
            print("S vector updated: {}".format(rewards))

            # Lists of rewards and actions(arms drawn)
            draw.append(next_action)
            rew.append(r)

        return rew, draw

    def TS(self, T):
        """
        Thomson Sampling algorithm
        :param T:
        :param MAB:
        :return:
        """

        number_draws, rewards = [0] * self.K, [0] * self.K
        rew = []
        draw = []

        for t in range(T):

            if self.expo:
                beta_seq = [ArmGamma(number_draws[i] + 0.5, rewards[i] + 1).sample() for i in range(self.K)]

            else:
                beta_seq = [ArmBeta(rewards[i] + 1, number_draws[i] - rewards[i] + 1).sample() for i in range(self.K)]

            print("betabinomial sequence  function from which we get the argmax: {}".format(beta_seq))
            # Get the argmax from the sequence function
            next_action = np.argmax(beta_seq)
            print("Next Arm to draw: {}".format(next_action + 1))

            next_arm = self.MAB[next_action]
            r = next_arm.sample()
            print("Reward of the next arm drawn: {}".format(r))

            # Updating the N(t) and S(t)
            number_draws[next_action] += 1
            print("N vector updated: {}".format(number_draws))

            rewards[next_action] += r
            print("S vector updated: {}".format(rewards))

            # Lists of rewards and actions(arms drawn)
            draw.append(next_action)
            rew.append(r)

        return rew, draw

    def regret(self, T, n, mode):

        rewards = [[]]
        p = max(self.means)
        t_vectore = (np.array(range(T)) + 1) * p

        if mode == 'UCB1':
            for i in range(n):
                print("iteration number {}".format(i))
                rew, draw = self.UCB1(T)
                cumulative_rew = np.cumsum(rew)
                rewards.append(cumulative_rew.tolist())

        elif mode == 'TS':
            for i in range(n):
                print("iteration number {}".format(i))
                rew, draw = self.TS(T)
                cumulative_rew = np.cumsum(rew)
                rewards.append(cumulative_rew.tolist())
        else:
            for i in range(n):
                print("iteration number {}".format(i))
                rew, draw = self.UCB1(T, naive=True)
                cumulative_rew = np.cumsum(rew)
                rewards.append(cumulative_rew.tolist())

        rewards_matrix = np.array(rewards[1:])
        expected_regret = t_vectore - np.mean(rewards_matrix, axis=0)

        return expected_regret

    def complexity(self):
        means = np.array(self.means)
        p = max(self.means)
        means = np.setdiff1d(means, p)
        _kl_ = np.array([kl(mean, p) for mean in np.setdiff1d(self.means, p)])

        _means = (means - p) / _kl_

        return np.sum(_means)

    def oracle(self, T):

        return [self.complexity() * log(i + 1) for i in range(T)]