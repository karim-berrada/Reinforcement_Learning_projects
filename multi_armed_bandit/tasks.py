import matplotlib.pyplot as plt

from multi_armed_bandit.modelisation import Game


K = 10
T = 200
n = 700
game = Game(K)

ucb_regrets = game.regret(T, n, mode="UCB1")
ts_regrets = game.regret(T, n, mode="TS")
naive_regrets = game.regret(T, n, mode="Naive")

def plot_rerets():
    plt.figure()
    plt.title('Regret curves')
    plt.xlabel("length of the game (T)")
    plt.ylabel('Expected regret')
    plt.plot(range(T), ucb_regrets, '-', label='UCB1 Algo')
    plt.plot(range(T), ts_regrets, 'r-', label='Thompson Sampling')
    plt.plot(range(T), naive_regrets, '-', label='Naive Algo')

    plt.legend()

    plt.show()
