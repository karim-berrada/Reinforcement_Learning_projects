import random

import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig

from MAB_RL_Advesarial.formalization import Simulator, EXP3
from MAB_RL_Advesarial.test import soda_strategy_discount, soda_strategy_nodiscount

##### Question 1 ######
gamma = 0.04
beta = 0.02
eta = gamma / (4)

G = [[2, -1], [0, 1]]
nbActions = len(G[0])


def EXP3vEXP3(n, eta, beta, G):

    player_A = EXP3(nbActions, eta, beta, gamma)
    player_B = EXP3(nbActions, eta, beta, gamma)

    ActionsA = []
    ActionsB = []
    Rew = []

    for i in range(n):

        action_A = player_A.play()
        ActionsA.append(action_A)

        action_B = player_B.play()
        ActionsB.append(action_B)

        reward_A = G[action_A - 1][action_B - 1]
        Rew.append(reward_A)

        player_A.get_reward(action_A, reward_A)
        player_B.get_reward(action_B, -reward_A)

    return ActionsA, ActionsB, Rew

n = 1000
count_A_1 =[]
count_B_1 =[]
reward_A =[]
ActionsA, ActionsB, Rew = EXP3vEXP3(n, eta, beta, G)

for i in range(1, n):
    Actions_A = ActionsA[:i]
    Actions_B = ActionsB[:i]
    Rewards = Rew[:i]
    count_A_1.append(float(Actions_A.count(1)) / float(len(Actions_A)))
    count_B_1.append(float(Actions_B.count(1)) / float(len(Actions_B)))
    reward_A.append(float(sum(Rewards)) / float(len(Rewards)))

### PLot the Nash Equilibria convergence ###
plt.figure()
plt.title('Convergence toward the Nash Equilibrium (0.5, 0.25)')
plt.ylabel('Proportion p of drawing arm 1 for player A and B')
plt.plot(range(n-1), count_A_1, '-', label='p_a_n')
plt.plot(range(n-1), count_B_1, '-', label='p_b_n')
plt.legend()

plt.show()
savefig('Convergence toward the Nash Equilibrium (0.5, 0.25).png')

### PLot the Game Value convergence ###
plt.figure()
plt.title('Convergence toward the Value of the Game Equilibrium (0.5).png')
plt.xlabel("Number of playing game iteration")
plt.ylabel('Expected Reward')
plt.plot(range(n-1), reward_A, '-', label='cum_sum reward')
plt.legend()

plt.show()
savefig('Convergence toward the Value of the Game Equilibrium (0.5).png')

##### Question 2 ######
# initializing the simulator
s = Simulator()

#  number of restocks
R = 5000

# parameters can be changed WITHIN the simulator object

# Trying the simulator

# print the number of drinks of each kind
print s.n_energy, s.n_nosugar

# perform a transition with no discount
rew, n_energy, n_nosugar = s.simulate(0)

# reset the simulator
s.reset()

# Comparison of one policy with the baseline

V = [0.1, 0.2, 0.5]
T = [2, 5, 10]


def policy1(n1, n2):
    return soda_strategy_discount(n1, n2)


def policy2(n1, n2):
    return soda_strategy_nodiscount(n1, n2)

# policy3=@(n1,n2) soda_strategy_param(n1,n2,T,V);


def policy3(n1, n2, T, V):

    diff = abs(n1 - n2)
    discount = V[0] * int((diff > T[0]) and (diff <= T[1])) \
               + V[1] * int((diff > T[1]) and (diff <= T[2])) \
               + V[2] * int(diff > T[2])
    if n1 > n2:
        return discount
    else:
        return - discount


def test_1_2(s, test=1, R=5000):
    """
    Test to compare the naive
    :param s:
    :param test:
    :param R:
    :return:
    """
    # nb of restocks to estimate the expectation
    tot_rewards = [0] * R
    tot_sodas = [0] * R

    for r in range(1, R + 1):

        # refill
        s.reset()
        rew, n_energy, n_nosugar = s.simulate(0)
        reward = rew
        t = 1
        while n_energy > 0 and n_nosugar > 0:
            # no re-fill is needed
            if test == 1:
                discount = policy1(n_energy, n_nosugar)
            else:
                discount = policy2(n_energy, n_nosugar)
            (rew, n_energy, n_nosugar) = s.simulate(discount)
            reward = reward + rew
            t = t + 1

        # total reward obtained
        tot_rewards[r - 1] = reward
        # number of sodas solde
        tot_sodas[r - 1] = t

    return tot_rewards, tot_sodas


def test_3(s, T, V, R=5000):

    # nb of restocks to estimate the expectation
    tot_rewards = [0] * R
    tot_sodas = [0] * R

    for r in range(1, R + 1):

        # refill
        s.reset()
        rew, n_energy, n_nosugar = s.simulate(0)
        reward = rew
        t = 1
        while n_energy > 0 and n_nosugar > 0:

            # no re-fill is needed
            discount = policy3(n_energy, n_nosugar, T, V)

            (rew, n_energy, n_nosugar) = s.simulate(discount)
            reward = reward + rew
            t = t+1

        # total reward obtained
        tot_rewards[r - 1] = reward
        # number of sodas solde
        tot_sodas[r - 1] = t

    return tot_rewards, tot_sodas

tot_rewards_1, tot_sodas_1 = np.cumsum(test_1_2(s, test=1, R=R)[0]), np.cumsum(test_1_2(s, test=1, R=R)[1])
tot_rewards_2, tot_sodas_2 = np.cumsum(test_1_2(s, test=2, R=R)[0]), np.cumsum(test_1_2(s, test=2, R=R)[1])
tot_rewards_3, tot_sodas_3 = np.cumsum(test_3(s, T, V, R=R)[0]), np.cumsum(test_3(s, T, V, R=R)[1])

_tot_rewards_1 = [tot_rewards_1[i] / float(i+1) for i in range(len(tot_rewards_1))]
_tot_rewards_2 = [tot_rewards_2[i] / float(i+1) for i in range(len(tot_rewards_1))]
_tot_rewards_3 = [tot_rewards_3[i] / float(i+1) for i in range(len(tot_rewards_1))]

plt.figure()
plt.title('Policy 1, 2 and 3 Cumulative Total Rewards ')
plt.xlabel("Simulation number")
plt.ylabel('Reward Total')
plt.plot(range(R), (tot_rewards_1), '-', label='tot_rewards_1')
plt.plot(range(R), (tot_rewards_2), '-', label='tot_rewards_2')
plt.plot(range(R), (tot_rewards_3), '-', label='tot_rewards_3')

plt.legend()

plt.show()
savefig('Cumulative Total Expected Reward obtained')

plt.figure()
plt.title('Policy 1, 2 and 3 Total Rewards ')
plt.xlabel("Simulation number")
plt.ylabel('Reward Total')
plt.plot(range(R), (_tot_rewards_1), '-', label='expected tot_rewards_1')
plt.plot(range(R), (_tot_rewards_2), '-', label='expected tot_rewards_2')
plt.plot(range(R), (_tot_rewards_3), '-', label='expected tot_rewards_3')

plt.legend()

plt.show()
savefig('Expected Total Reward obtained')

plt.figure()
plt.title('Policy 1, 2 and 3 Cumulative Total Sodas ')
plt.xlabel("Simulation number")
plt.ylabel('Soda Sold Total')
plt.plot(range(R), (tot_sodas_1), '-', label='tot_sodas_1')
plt.plot(range(R), (tot_sodas_2), '-', label='tot_sodas_2')
plt.plot(range(R), (tot_sodas_3), '-', label='tot_sodas_3')

plt.legend()

plt.show()
savefig('Cumulative Total Soda Sold')


# Choose a bandit problem

def DrawArm(T, V, s):

    tot_rewards_3, tot_sodas_3 = test_3(s, T, V, R=1)

    return tot_rewards_3[0]


def generate_tables(nb_arms):

    TableT = [sorted(random.sample(range(50),  3)) for j in range(nb_arms)]
    TableV = [sorted([random.random() for i in range(3)]) for j in range(nb_arms)]

    return TableT, TableV


class game:

    def __init__(self, nb_arms, TableV, TableT, s):

        self.s = s
        self.nb_arms = nb_arms
        self.TableV = TableV
        self.TableT = TableT

    def draw_arm(self, T, V):

        return DrawArm(T, V, self.s) / 100.

    def exp3(self, R, eta, beta, gamma):

        nbActions = self.nb_arms
        actor = EXP3(nbActions, eta, beta, gamma)

        Actions= []
        Rewards = []
        TableT, TableV = self.TableT, self.TableV
        for i in range(R):

            action = actor.play()
            Actions.append(actor.play())

            rew = self.draw_arm(TableT[action - 1], TableV[action - 1])
            Rewards.append(rew)

            actor.get_reward(action, rew)

        return Actions, Rewards

    def UCB1(self, R, naive=False):

        """
        The UCB1 algorithm starts with an initialization phase that draws each arm once, and then for each t more than K,
        it chooses the arm to draw at t+1 according to a optimal function to maximize
        :param T:
        :param MAB:
        :return:
        """
        # Initialization

        rewards = [self.draw_arm(self.TableT[i], self.TableV[i]) for i in range(self.nb_arms)]
        number_draws = np.array([1] * self.nb_arms)

        rew = []
        draw = []

        for t in range(R):
            print("len = {}".format(t))
            if naive:
                opt_func = rewards / number_draws
            else:
                opt_func = rewards / number_draws + np.sqrt(np.log(t + 1) / (2. * number_draws))
            print("optimization function from which we get the argmax: {}".format(opt_func))

            # Get the argmax from the optimization function
            next_action = np.argmax(opt_func)
            print("Next Arm to draw: {}".format(next_action + 1))

            r = self.draw_arm(self.TableT[next_action], self.TableV[next_action])
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

nb_arms = 5
R = 5000
TableT, TableV = generate_tables(nb_arms)
gamma = 0.04
beta = 0.02
eta = gamma / (4)

mab_pbm = game(nb_arms, TableV, TableT, s)

# UCB1 AND EXP3 Results
exp_actions, exp_rewards = mab_pbm.exp3( R, eta, beta, gamma)
ucb_rewards, ucb_actions = mab_pbm.UCB1(R)


unique, counts = np.unique(ucb_actions, return_counts=True)
print counts
# The best arm is the one chosen the more often by the UCB1 Algorithm
best = np.argmax(counts)
bestV, bestT = mab_pbm.TableV[best], mab_pbm.TableT[best]
best_rewards = [mab_pbm.draw_arm(bestT, bestV) for i in range(R)]

plt.figure()
plt.title('Rewards')
plt.xlabel("Simulation number")
plt.ylabel('Reward expected')
plt.plot(range(R), np.cumsum(exp_rewards) / np.arange(1, R + 1, dtype="float"), '-', label='EXP Rewards expected')
plt.plot(range(R), np.cumsum(ucb_rewards) / np.arange(1, R + 1, dtype="float"), '-', label='UCB1 Rewards expected')
plt.plot(range(R), np.cumsum(best_rewards) / np.arange(1, R + 1, dtype="float"), '-', label='Best Rewards expected')

plt.legend()

plt.show()
savefig('UCB_EXP_expected_Rewards ')