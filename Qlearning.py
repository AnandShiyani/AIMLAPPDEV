# imports required packages
import pandas as pd
import numpy as np
def get_possible_next_states(state, F, states_count):
    possible_next_states = []
    for i in range(states_count):
        if F[state, i] == 1:
            possible_next_states.append(i)
    return possible_next_states
def get_random_next_state(state, F, states_count):
    possible_next_states = get_possible_next_states(state, F, states_count)
    next_state = possible_next_states[np.random.randint(0, len(possible_next_states))]
    return next_state
# Loads up feasibility matrix denoted as F
F = np.loadtxt("feasibility_matrix.csv", dtype="int", delimiter=',')
# Prints the matrix
print(F)
# Loads up reward matrix denoted as R
R = np.loadtxt("reward_matrix.csv", dtype="float", delimiter=',')
# Prints the matrix; Note that reward is provided only when
# goal is reached. Refer location (9, 14).
print(R)
# Initializes quality matrix, denoted by Q, with all zeros
Q = np.zeros(shape=[15,15], dtype=np.float32)
# Prints the Q matrix
display(pd.DataFrame(Q, dtype=float).style.format(precision=2))
def train(F, R, Q, gamma, lr, goal_state, states_count, episodes):
    for i in range(0, episodes):
# Selects a random start state
        current_state = np.random.randint(0, states_count)
# Continues till goal state is reached
        while(True):
# Selects a random next state from the current state
            next_state = get_random_next_state(current_state, F, states_count)
# Gets all possible states from that next state
            possible_next_next_states = get_possible_next_states(next_state, F, states_count)
# Compares the Q value between two possible next states
            max_Q = -9999.99
            for j in range(len(possible_next_next_states)):
                next_next_state = possible_next_next_states[j]
                q = Q[next_state, next_next_state]
                if q > max_Q:
                    max_Q = q
# Updates the Q value using Bellman equation [refer maze image caption]
            Q[current_state][next_state] = \
                                         ((1 - lr) * Q[current_state][next_state]) + (lr * (R[current_state][next_state] + (gamma *max_Q)))
# Changes state by considering next state as current state and
# the training continues till goal state is reached
            current_state = next_state
            if current_state == goal_state:
                break

# Sets hyperparameters
gamma = 0.5 # discount factor
lr = 0.5 # learning_rate
goal_state = 14
states_count = 15
episodes = 1000
np.random.seed(42)
# starts trainingtrain(F, R, Q, gamma, lr, goal_state, states_count, episodes)
# Prints Q matrix generated out of training

display(pd.DataFrame(Q, dtype=float).style.format(precision=2))
# The function walk()
def print_shortest_path(start_state, goal_state, Q):
    current_state = start_state
    print(str(current_state) + "->", end="")
# Loops till goal is reached and keeps on tracing the path
    while current_state != goal_state:
# Chooses the best state from possible states and keep on printing
        next_state = np.argmax(Q[current_state])
        print(str(next_state) + "->", end="")
# Considers next state as current state and continiues till goal is reached
        current_state = next_state
    print("Goal Reached.\n")
# Performs few tests for agent to get the shortest path
start_state = 8
print("Best path to reach goal from state {0} to goal state {1}".format(start_state, goal_state))
print_shortest_path(start_state, goal_state, Q)
start_state = 13
print("Best path to reach goal from state {0} to goal state {1}".format(start_state, goal_state))
print_shortest_path(start_state, goal_state, Q)
start_state = 6
print("Best path to reach goal from state {0} to goal state {1}".format(start_state, goal_state))
print_shortest_path(start_state, goal_state, Q)
start_state = 1
print("Best path to reach goal from state {0} to goal state {1}".format(start_state, goal_state))
print_shortest_path(start_state, goal_state, Q)
