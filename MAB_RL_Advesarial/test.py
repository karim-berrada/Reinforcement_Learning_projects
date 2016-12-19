from MAB_RL_Advesarial.formalization import Simulator

s = Simulator()

R = 5000  # nb of restocks to estimate the expectation
tot_rewards = [0] * R
tot_sodas = [0] * R


def policy(n1, n2):

    return soda_strategy_discount(n1, n2)


def soda_strategy_nodiscount( n_energy, n_nosugar):

    discount = 0.0

    return discount


def soda_strategy_discount(n_energy, n_nosugar):

    if n_energy <= n_nosugar:
        discount = -0.2
    else:
        discount = 0.2

    return discount


for r in range(1, R + 1):

    # refill
    s.reset()
    rew, n_energy, n_nosugar = s.simulate(0)
    reward = rew
    t = 1
    while n_energy > 0 and n_nosugar > 0:

        # no re-fill is needed
        discount = policy(n_energy, n_nosugar)

        [rew, n_energy, n_nosugar] = s.simulate(discount)
        print(rew)
        reward = reward + rew
        t = t+1

    # total reward obtained
    tot_rewards[r - 1] = reward
    # number of sodas solde
    tot_sodas[r - 1] = t

