import matplotlib.pyplot as plt
from matplotlib import interactive
import numpy as np
from tqdm import tqdm
from plot_utils import create_plot, plotter
from RL_utils import return_a_random_policy
from RL_utils import step
from RL_utils import choose_an_action_based_on_pi
from RL_utils import return_epsilon_greedy_pi
from RL_utils import initialize_the_state
from RL_utils import Bellmann_iteration

# 1. Initialization

# 1.1 inputs and the random seed
np.random.seed(0)

# 1.2 grid size along each direction
n = 8

# 1.3 nr of actions
# if you are going to change it:
# change the pointwise action function in RL_utils as well
nr_actions = 4

# 1.4 discount
# should be set to a value smaller than 1
gamma = 0.98

# 1.5 nr of learning episodes
nr_episodes = 10_000

# 1.6 epsilon for the epsilon greedy
epsilon = 0.4

# 1.7 policy $\pi$
# initializing policy to a random policy
# initializing Q to zero

pi = return_a_random_policy(n, nr_actions)
Q_accumulate = np.zeros((n, n, nr_actions))
Q_visit_counter = np.zeros((n, n, nr_actions))

# 1.8 setting up the plot
ax = create_plot(n)
plt.ion()
interactive(True)
plt.cla()
ax.axis('off')


# 2. policy iteration
for episode_id in range(nr_episodes):
    print(episode_id)

    # 2.1 policy evaluation loops
    for _ in tqdm(range(nr_actions * n * n)):
        terminated = False
        s0 = initialize_the_state(n)
        a0 = choose_an_action_based_on_pi(s0, pi)
        state, reward, terminated = step(s0, a0, n)
        G = reward
        gamma_power_n = gamma
        while not terminated:
            action_id = choose_an_action_based_on_pi(state, pi)
            new_state, reward, terminated = step(state, action_id, n)
            G += gamma_power_n * reward
            state = new_state + 0.0
            gamma_power_n *= gamma

        i, j = s0[0]
        Q_accumulate[i, j, a0] += G
        Q_visit_counter[i, j, a0] += 1.0

        Q = Q_accumulate / (Q_visit_counter + 0.01)

    # 2.2 updating the policy using Q values (to epsilon greedy)
    pi = return_epsilon_greedy_pi(Q, epsilon)

    v = np.zeros(shape=(n, n))

    # 2.3 visual monitoring porpuses
    for i in range(100):
        v = Bellmann_iteration(np.argmax(pi, axis=-1), v, gamma)
    plotter(ax, v)
