import numpy as np
from numpy import exp, log
from numpy.random import random, beta


class ArmBernoulli:
    """Bernoulli arm"""

    def __init__(self, p):
        """
        p: Bernoulli parameter
        """
        self.p = p
        self.mean = p
        self.var = p * (1 - p)

    def sample(self):
        reward = random() < self.p
        if reward:
            return 1.
        else:
            return 0.


class ArmBeta:
    """arm having a Beta distribution"""

    def __init__(self, a, b):
        """
        a: first beta parameter
        b: second beta parameter
        """
        self.a = a
        self.b = b
        self.mean = a / (a + b)
        self.var = (a * b) / ((a + b) ** 2 * (a + b + 1))

    def sample(self):
        reward = beta(self.a, self.b)
        return reward


class ArmExp():
    """arm with trucated exponential distribution"""

    def __init__(self, lambd):
        """
        lambd: parameter of the exponential distribution
        """
        self.lambd = lambd
        self.mean = (1 / lambd) * (1 - exp(-lambd))
        self.var = 1  # compute it yourself!

    def sample(self):
        reward = min(-1 / self.lambd * log(random()), 1)
        return reward


def simu(p):
    """
    draw a sample of a finite-supported distribution that takes value
    k with porbability p(k)
    p: a vector of probabilities
    """
    q = p.cumsum()
    u = random()
    i = 0
    while u > q[i]:
        i += 1
        if i >= len(q):
            raise ValueError("p does not sum to 1")
    return i


class ArmFinite:
    """arm with finite support"""

    def __init__(self, X, P):
        """
        X: support of the distribution
        P: associated probabilities
        """
        self.X = np.array(X)
        self.P = np.array(P)
        self.mean = (self.X * self.P).sum()
        self.var = (self.X ** 2 * self.P).sum() - self.mean ** 2

    def sample(self):
        i = simu(self.P)
        reward = self.X[i]
        return reward
