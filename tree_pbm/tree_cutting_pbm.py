import matplotlib.pyplot as plt

import numpy as np


class Tree:
    'Common base class for all trees'

    def __init__(self, h=1, max_h=10, s="sane"):

        if h > max_h:
            raise Exception("height cannot be superior to max height")
        else:
            self.height = h
            self.max_height = max_h
            self.health_state = s
            self.maintainance_cost = 1
            self.plantation_cost = 3
            self.unit_value = 3
            self.sickness_proba = np.random.uniform(low=0., high=0.3, size=(self.max_height,))


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

        if state % 2 == 0:
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
            self.reward = - self.tree.maintainance_cost
            self.tree.is_sick(state)
            if self.tree.health_state != "sick":
                self.tree.height = self.get_transition(state, proba_matrix)
            else:
                self.reward -= self.tree.plantation_cost
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

    def naive_proba_matrix(self):

        naive_matrix = []
        for i in range(1, self.tree.max_height + 1):
            state_a = i
            action = self.naive_policy(state_a)
            p = self.proba_matrix(action)
            naive_matrix += [p[state_a]]

        return naive_matrix

    def proba_matrix(self, action):

        if action == "cut":
            l = [1] + [0] * (self.tree.max_height - 1)
            p = [l] * (self.tree.max_height)
        else:
            p = [
                [self.transition_proba(i, j) for j in range(1, self.tree.max_height + 1)]
                for i in range(1, self.tree.max_height + 1)]
        return p

    def get_transition(self, state, proba_matrix):

        number_of_states = len(proba_matrix)
        probabilities = proba_matrix[state]
        new_state = np.random.choice(number_of_states, 1, p=probabilities)[0]
        return new_state


mdp = tree_MDP()

#def check_mdp_tree(mdp):
matrice = mdp.proba_matrix("no")
for i in range(len(matrice)):
    print(sum(matrice[i]))


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

        # print("starting traj number {}".format(number_traj))
        while r_max * (gamma ** time) > 0.01:
            action = mdp.naive_policy(state)
            proba_matrix = mdp.proba_matrix(action)
            new_state, reward = mdp.tree_sim(state, action, proba_matrix)
            traj_return += reward * (gamma ** time)
            time += 1
            state = new_state
        # print("finished traj number {}".format(number_traj))
        all_traj_return += traj_return
        number_traj += 1

    return float(all_traj_return), float(number_traj)

def mc_value_func(mdp):

    n_states = mdp.tree.max_height
    value_func_estimation = {}
    for init_state in range(n_states + 1):
        value_func_estimation[init_state] = monte_carlo_eval(mdp, n=5000, init_state=init_state)

    return value_func_estimation

mc_evals = []
n_trajectories = []
eval = 0
n_traj = 0
number_run = 5
for n in range(1, 500):

    all_return, number_traj = monte_carlo_eval(mdp, n=number_run)
    eval += all_return
    n_traj += number_run
    mc_evals += [eval / n_traj]
    n_trajectories += [n_traj ]
    print("estimation with {} traj done".format(n_traj))
print("Done !")

xlabel = "number of trajectories"
ylabel = "Estimated Value function(state = 1, naive_policy) "
x = n_trajectories
y = mc_evals
plt.plot(x, y, '-')
plt.ylabel(ylabel)
plt.xlabel(xlabel)
plt.show()

