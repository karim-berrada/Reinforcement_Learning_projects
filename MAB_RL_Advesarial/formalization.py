import random as random
from math import exp, log

import numpy as np
from numpy import cumsum
from numpy.random import choice

rand = random.uniform(0,1)


class EXP3:

    def __init__(self, nbActions, eta, beta, gamma):

        self.nbActions = nbActions
        self.eta = eta
        self.gamma = gamma
        self.beta = beta
        self.w = [1.] * self.nbActions

    def initialize(self):

        self.w = [1.] * self.nbActions

    def play(self):
        # use the weights to choose an action to play
        W = float(sum(self.w))
        print("weights for exp3 : {}".format(self.w))
        K = self.nbActions
        P = [(1 - self.gamma) * self.w[i] / W + self.gamma / K for i in range(K)]

        list_actions = np.arange(1, K + 1)
        action = choice(list_actions, p=P)

        return action

    def get_reward(self, arm, reward):

        # % update the weights given the arm drawn and the reward observed
        W = float(sum(self.w))
        K = self.nbActions
        P = [(1 - self.gamma) * self.w[i] / W + self.gamma / K for i in range(K)]
        print('proba matrice : {}'.format(P))
        X_hat = float(reward) / float(P[arm - 1])
        X_tilda = [float(self.beta) / float(P[i]) for i in range(K)]
        X_tilda[arm - 1] += X_hat

        # Updating the weights
        w = self.w
        for i in range(K):
            w[i] *= exp(self.eta * X_tilda[i])
        self.w = w


class Simulator:

    def __init__(self, state=0, p_exam_noexam=0.7, p_noexam_exam=0.3, std_price=1.0, n_energy=50., n_nosugar=50.):

        self.state = state # {0=no-exam, 1=exam}
        self.p_exam_noexam = p_exam_noexam # probability of transition from an exam to no-exam state
        self.p_noexam_exam = p_noexam_exam # probability of transition from an no_exam to exam state
        self.std_price = std_price  # the price of either soda when no discount is applied
        self.n_energy = n_energy # number of energy drink cans
        self.n_nosugar = n_nosugar  # number of no sugar cans

    def reset(self):

        self.n_energy = 50
        self.n_nosugar = 50
        self.state = 0

    def simulate(self, discount):
        discount_fraction = float(discount) / float(self.std_price)
        # self.state = 0;
        if self.state == 0:
            # no-exam situation
            # pref_energy = rand;
            pref_energy = 0.6
            pref_nosugar = 1 - pref_energy

            # apply changes depending on the discount
            if discount_fraction > 0.0:
                # the energy drink is discounted
                pref_energy *= exp(2 * discount_fraction * log(1. / pref_energy))
                pref_nosugar = 1 - pref_energy

            elif discount_fraction < 0.0:
                # the sugar free is discounted
                pref_nosugar *= exp(-2 * discount_fraction * log(1. / pref_nosugar))
                pref_energy = 1 - pref_nosugar
        elif self.state == 1:
            # exam situation

            pref_energy = 0.8
            pref_nosugar = 1 - pref_energy

            # Apply changes depending on the discount
            if discount_fraction > 0.0:
                # the energy drink is discounted

                if 4 * pref_energy > 1.0:

                    pref_energy += (1 - pref_energy) * pref_energy ** 4
                else:
                    pref_energy += 3 * pref_energy * pref_energy ** 4

                pref_nosugar = 1 - pref_energy

            elif discount_fraction < 0.0:
                #  the sugar free is discounted
                if 4 * pref_nosugar > 1.0:
                    pref_nosugar += (1 - pref_nosugar) * pref_nosugar ** 4

                else:
                    pref_nosugar += 3 * pref_nosugar * pref_nosugar ** 4

                pref_energy = 1 - pref_nosugar

        # random user preference
        r_ = random.uniform(0, 1)
        if r_ < pref_energy:
            # user with preference for energy drink
            reward = self.std_price - max(discount, 0)
            self.n_energy = self.n_energy - 1
        else:
            # user with preference for energy drink
            reward = self.std_price + min(discount, 0)
            self.n_nosugar = self.n_nosugar - 1

        # evolution of the state of the environment
        if self.state == 0 and r_ < self.p_noexam_exam:
            self.state = 1
        elif self.state == 1 and r_ < self.p_exam_noexam:
            self.state = 0

        n_energy = self.n_energy
        n_nosugar = self.n_nosugar

        return reward, n_energy, n_nosugar


def simu(p):

    q = cumsum(p)
    u = rand
    i = 1
    while u > q(i):
        i += 1

    return i







