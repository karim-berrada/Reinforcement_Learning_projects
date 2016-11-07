import matplotlib.pyplot as plt
import copy

import numpy as np


### 1 - Formalisation ###
class Tree:
    'Common base class for all trees'

    def __init__(self, h=1, max_h=8, s="sane"):

        if h > max_h:
            raise Exception("height cannot be superior to max height")
        else:
            self.height = h
            self.max_height = max_h
            self.health_state = s
            self.maintainance_cost = 1
            self.plantation_cost = 3
            self.unit_value = 3
            self.sickness_proba = np.random.uniform(low=0., high=0.1, size=(self.max_height + 1,))

    def reinitialize_tree(self):
        self.height = 1
        self.health_state = "sane"

    def is_sick(self, state):
        get_sick_proba = self.sickness_proba[state]
        sick = np.random.choice(2, 1, p=[get_sick_proba, 1 - get_sick_proba])[0]
        if sick == 0:
            self.health_state = "sick"


class tree_MDP:

    def __init__(self):

        self.tree = Tree()
        self.reward = 0
        self.interest_rate = 0.05
        self.gamma = 1 / (self.interest_rate + 1)

    def naive_policy(self, state):

        if state % 4 == 0:
            action = "cut"
        else:
            action = "no cut"
        return action

    def tree_sim(self, state, action, proba_matrix):

        self.tree.height = state
        if action == "cut":
            self.reward = self.tree.height * self.tree.unit_value - self.tree.plantation_cost
            self.tree.reinitialize_tree()

        else:
            if state == 1:
                self.reward = - self.tree.plantation_cost
            else:
                self.reward = - self.tree.maintainance_cost
            self.tree.is_sick(state)
            if self.tree.health_state != "sick":
                self.tree.height = self.get_transition(state, proba_matrix)
            else:
                self.tree.reinitialize_tree()
        new_state = self.tree.height
        reward = self.reward
        self.reward = 0
        return new_state, reward

    def transition_proba(self, state_a, state_b):

        if state_b == 1:
            # the tree is sick and dies so it goes back to state 1
            proba = self.tree.sickness_proba[state_a - 1]
        elif state_b <= state_a:
            if state_b == state_a and state_a == self.tree.max_height:
                proba = float(1 - self.tree.sickness_proba[state_a - 1])
            else:
                proba = 0
        else:
            # the tree is not sick and have the same chance to reach any state after state_a
            proba = float(1 - self.tree.sickness_proba[state_a - 1]) / float(self.tree.max_height - state_a)
        return proba

    def naive_proba_rewards(self):

        naive_matrix = []
        naive_reward = []
        for i in range(1, self.tree.max_height + 1):
            state_a = i
            action = self.naive_policy(state_a)
            p = self.proba_matrix(action)
            sim, reward = self.tree_sim(state_a, action, p)

            naive_matrix += [p[state_a - 1]]
            naive_reward += [reward]

        return naive_matrix, naive_reward

    def proba_matrix(self, action):

        if action == "cut":
            max_height = self.tree.max_height
            l = [1] + [0] * (max_height  - 1)
            p = [l] * (max_height)
        else:
            p = [
                [self.transition_proba(i, j) for j in range(1, self.tree.max_height + 1)]
                for i in range(1, self.tree.max_height + 1)]
        return p

    def get_transition(self, state, proba_matrix):

        number_of_states = len(proba_matrix)
        probabilities = proba_matrix[state - 1]
        new_state = np.random.choice(number_of_states, 1, p=probabilities)[0] + 1
        return new_state


# Cheking the MDP tree probabilities
mdp = tree_MDP()

matrice = mdp.proba_matrix("no cut")
for i in range(len(matrice)):
    print(sum(matrice[i]))


### 2 - Policy Evaluation ###

def monte_carlo_eval(mdp, n=1000, init_state=1):
    tree = mdp.tree
    r_max = tree.max_height * tree.unit_value - tree.plantation_cost
    gamma = mdp.gamma
    number_traj = 0
    all_traj_return = 0
    for i in range(n):
        time = 0
        state = init_state
        traj_return = 0

        while r_max * (gamma ** time) > 0.01:
            action = mdp.naive_policy(state)
            proba_matrix = mdp.proba_matrix(action)
            new_state, reward = mdp.tree_sim(state, action, proba_matrix)
            traj_return += reward * (gamma ** time)
            time += 1
            state = new_state

        all_traj_return += traj_return
        number_traj += 1

    return float(all_traj_return), float(number_traj)


# Monte Carlo
def mc_final_eval(mdp, number_run, init_state):
    """

    :param mdp: Class
    :param number_run: Number of iteration
    :param init_state: integer (state)
    :return: integer and array with the set of values given all the MC trajectories
    """
    mc_evals = []
    n_trajectories = []
    eval = 0
    n_traj = 0
    n_run = 5
    for n in range(1, number_run / n_run):

        all_return, number_traj = monte_carlo_eval(mdp, n=n_run, init_state=init_state)
        eval += all_return
        n_traj += n_run
        mc_evals += [eval / n_traj]
        n_trajectories += [n_traj]
        print("estimation with {} traj done".format(n_traj))
    print("Done !")

    return n_trajectories, mc_evals


def mc_value_func(mdp, number_run):

    n_states = mdp.tree.max_height
    value_func_estimation = {}
    for init_state in range(1, n_states + 1):

        n_trajectories, mc_evals = mc_final_eval(mdp, number_run, init_state)

        value_func_estimation[init_state] = mc_evals[-1]
        print(init_state, value_func_estimation[init_state])

    return value_func_estimation


# Dynamic Programming
def dynamic_value_func(mdp):
    """

    :param mdp: Class
    :return: Vector of the Values for each state
    """
    naive_matrix, naive_reward = mdp.naive_proba_rewards()
    naive_matrix = np.array(naive_matrix)
    n = len(naive_reward)
    id = np.eye(n)
    gamma = mdp.gamma

    return np.linalg.inv((id - gamma * naive_matrix)).dot(np.transpose(naive_reward))


# Plots
def plot_mc_sim(mdp, number_run, init_state):

    n_trajectories, mc_evals = mc_final_eval(mdp, number_run, init_state)
    n = len(mc_evals)
    dp_value = dynamic_value_func(mdp)
    dp_value = dp_value[0]
    dp_value = np.array([dp_value] * n)
    mc_evals = np.array(mc_evals)

    xlabel = "number of trajectories"
    ylabel = "Estimated Value function(state = 5, naive_policy) - Exact Value function(Dynamic Programming)"
    x = n_trajectories
    y = mc_evals - dp_value
    plt.plot(x, y, '-')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()


### 3 - Optimal Policy ###

# Value iteration
def Value_Iteration(mdp, epsilon):
    """

    :param mdp: Class
    :param epsilon: L2 norm difference between the successiv vectors threshold
    :return: Value vector, Optimal Policy, All the sets of Values of the VI
    """
    Values_V = []
    max_height = mdp.tree.max_height
    gamma = mdp.gamma
    V = np.zeros(max_height)
    policy = np.zeros(max_height)
    all_states = range(1, mdp.tree.max_height + 1)
    V_prec= V + 1
    number_of_iteration = 0
    while np.linalg.norm(V - V_prec) > epsilon:

        V_prec = copy.copy(V)
        Values_V.append(V_prec)

        for state in all_states:
            table = [mdp.tree_sim(state, action, mdp.proba_matrix(action))[1] +
                     gamma * (np.array(mdp.proba_matrix(action)).dot(V_prec))[state - 1]
                     for action in ["cut", "no_cut"]]
            V[state - 1] = np.array(table).max(axis=0)
        number_of_iteration += 1
        print("Computing VI for iteration number {}".format(number_of_iteration))

    for state in all_states:
        table = [mdp.tree_sim(state, action, mdp.proba_matrix(action))[1] +
                 gamma * (np.array(mdp.proba_matrix(action)).dot(V))[state - 1]
                 for action in ["cut", "no_cut"]]
        policy[state - 1] = np.argmax(np.array(table), axis=0)

    policy = ["cut" if x == 0 else "no_cut" for x in policy]

    return V, policy, Values_V


def get_proba_reward(mdp, Policy):

    matrix = []
    reward_vector = []

    for i in range(1, mdp.tree.max_height + 1):
        state_a = i
        action = Policy[state_a - 1]
        if action == 0:
            action = "cut"
        else:
            action = "no_cut"

        p = mdp.proba_matrix(action)
        sim, reward = mdp.tree_sim(state_a, action, p)

        matrix += [p[state_a - 1]]
        reward_vector += [reward]

    return matrix, reward_vector


# Policy iteration
def Policy_Iteration(mdp):

    Values_V = []

    max_height = mdp.tree.max_height
    gamma = mdp.gamma
    V = np.zeros(max_height)
    policy = np.zeros(max_height)
    all_states = range(1, mdp.tree.max_height + 1)
    for state in all_states:
        action = mdp.naive_policy(state)
        if action == "cut":
            policy[state - 1] = 0
        else:
            policy[state - 1] = 1
    V_prec = V + 1
    number_of_iteration = 0

    while (V != V_prec).any():

        V_prec = copy.copy(V)
        Values_V.append(V_prec)

        matrix, reward_vector = get_proba_reward(mdp, policy)
        matrix = np.array(matrix)
        n = len(reward_vector)
        id = np.eye(n)

        V = np.linalg.inv((id - gamma * matrix)).dot(np.transpose(reward_vector))

        for state in all_states:
            table = [mdp.tree_sim(state, action, mdp.proba_matrix(action))[1] +
                     gamma * (np.array(mdp.proba_matrix(action)).dot(V))[state - 1]
                     for action in ["cut", "no_cut"]]
            policy[state - 1] = np.argmax(np.array(table), axis=0)

        number_of_iteration += 1
        print("Computing VI for iteration number {}".format(number_of_iteration))

    policy = ["cut" if x == 0 else "no_cut" for x in policy]
    return V, policy, Values_V


def study_convergence(mdp, epsilon=0.01):

    vi_values = Value_Iteration(mdp, epsilon)
    pi_valies = Policy_Iteration(mdp)
    Value_list_VI = [np.linalg.norm(vi_values[0] - V) for V in vi_values[2]]
    Value_list_PI = [np.linalg.norm(pi_valies[0] - V) for V in pi_valies [2]]

    plt.figure()

    plt.title('Convergence of Value Iteration and Policy iteration')
    plt.xlabel("Number of Iteration")
    plt.ylabel('Norm of Vn - V')
    plt.plot(range(len(Value_list_VI)), Value_list_VI, '-', label='Value Iteration')
    plt.plot(range(len(Value_list_PI)), Value_list_PI, 'r-', label='Policy Iteration')

    plt.legend()

    plt.savefig("Convergence_of_Policy_Iteration_and_Value_Iteration.jpg")

    plt.show()
