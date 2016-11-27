import matplotlib.pyplot as plt
import numpy as np

from multi_armed_bandit.modelisation import Game


K = 4
T = 5000
n = 1000

perso_means_a = [0.01, 0.05, 0.1, 0.9]
perso_means_b = [0.4, 0.42, 0.44, 0.46]

game = Game(10)
game_a = Game(K, perso=True, perso_means=perso_means_a)
game_b = Game(K, perso=True, perso_means=perso_means_b)

ucb_regrets = game.regret(T, n, mode="UCB1")
ts_regrets = game.regret(T, n, mode="TS")
naive_regrets = game.regret(T, n, mode="Naive")

# plotting with random means:
plt.figure()
plt.title('Regret curves (Excpected number with n={} iteration'.format(n))
plt.xlabel("length of the game (T)")
plt.ylabel('Expected regret')
plt.plot(range(T), ucb_regrets, '-', label='UCB1 Algo')
plt.plot(range(T), ts_regrets, 'r-', label='Thompson Sampling')
plt.plot(range(T), naive_regrets, '-', label='Naive Algo')

plt.legend()

plt.show()

# Comparison between two arm bandits
oracle_a = game_a.oracle(T)
oracle_a = -1 * np.array(oracle_a)
ucb_regrets_a = game_a.regret(T, n, mode="UCB1")
ts_regrets_a = game_a.regret(T, n, mode="TS")

oracle_b = game_b.oracle(T)
oracle_b = -1 * np.array(oracle_b)
ucb_regrets_b = game_b.regret(T, n, mode="UCB1")
ts_regrets_b = game_b.regret(T, n, mode="TS")

# plotting the two bandits problems:
plt.figure()
plt.title('Regret curves for small complexity problem')
plt.xlabel("length of the game (T)")
plt.ylabel('Expected regret')
plt.plot(range(T), ucb_regrets_a, '-', label='UCB1 Algo')
plt.plot(range(T), ts_regrets_a, 'r-', label='Thompson Sampling')
plt.plot(range(T), oracle_a, '-', label='Oracle')

plt.legend()

plt.show()

plt.figure()
plt.title('Regret curves for large complexity problem')
plt.xlabel("length of the game (T)")
plt.ylabel('Expected regret')
plt.plot(range(T), ucb_regrets_b, '-', label='UCB1 Algo')
plt.plot(range(T), ts_regrets_b, 'r-', label='Thompson Sampling')
plt.plot(range(T), oracle_b, '-', label='Oracle')

plt.legend()

plt.show()

#

K = 4
T = 5000
n = 1000

exp_game = Game(K, expo=True)
means = exp_game.means

# Means that we have thanks to the Gamma(0.5, 1) prior
print(means)

ucb_regrets = exp_game.regret(T, n, mode="UCB1")
ts_regrets = exp_game.regret(T, n, mode="TS")

plt.figure()
plt.title('Regret curves for non parametric problem')
plt.xlabel("length of the game (T)")
plt.ylabel('Expected regret')
plt.plot(range(T), ucb_regrets_b, '-', label='UCB1 Algo')
plt.plot(range(T), ts_regrets_b, 'r-', label='Thompson Sampling')

plt.legend()

plt.show()