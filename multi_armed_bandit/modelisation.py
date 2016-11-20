from random import random

import numpy as np

from multi_armed_bandit.formalization import ArmBernoulli, ArmBeta, ArmExp, ArmFinite

K = 10

means = [random() for i in range(K)]
MAB = [ArmBernoulli(means[i]) for i in range(K)]


def initialize(MAB):

    rewards = [arm.sample() for arm in MAB]
    number_draws = np.array([1] * K)

    return number_draws, rewards


def UCB1(T, MAB):

    """
    The UCB1 algorithm starts with an initialization phase that draws each arm once, and then for each t more than K,
    it chooses the arm to draw at t+1 according to a optimal function to maximize
    :param T:
    :param MAB:
    :return:
    """
    # Initialization
    number_draws, rewards = initialize(MAB)
    rew = []
    draw = []

    for t in range(T):

        opt_func = rewards / number_draws + np.sqrt(np.log(t + 1) / (2. * number_draws))
        print("optimization function from which we get the argmax: {}".format(opt_func))

        # Get the argmax from the optimization function
        next_action = np.argmax(opt_func)
        print("Next Arm to draw: {}".format(next_action + 1))

        next_arm = MAB[next_action]
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


def TS(T, MAB):
    """
    Thomson Sampling algorithm
    :param T:
    :param MAB:
    :return:
    """

    number_draws, rewards = [0] * K, [0] * K
    rew = []
    draw = []

    for t in range(T):

        beta_seq = [ArmBeta(rewards[i] + 1, number_draws[i] - rewards[i] + 1).sample() for i in range(K)]
        print("betabinomial sequence  function from which we get the argmax: {}".format(beta_seq))
        # Get the argmax from the betabinomial sequence function
        next_action = np.argmax(beta_seq)
        print("Next Arm to draw: {}".format(next_action + 1))

        next_arm = MAB[next_action]
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
