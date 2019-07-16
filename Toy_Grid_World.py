import numpy as np


def create_episodes():

    first_over = [[1, 0], 1, [2, 0], 2, [2, 1], 2, [2, 2], 3, [1, 2], 2, [1, 3], 2]
    second_over = [[1, 4], 1, [2, 4], 2, [2, 5], 2, [2, 6], 3]

    first_under = [[1, 0], 3, [0, 0], 2, [0, 1], 2, [0, 2], 3, [1, 2], 2, [1, 3], 2]
    second_under = [[1, 4], 3, [0, 4], 2, [0, 5], 2, [0, 6], 1]

    episodes = [[first_over, second_over],
                [first_over, second_over],
                [first_over, second_over],
                [first_over, second_over],
                [first_over, second_over],
                [first_over, second_under],
                [first_over, second_under],
                [first_over, second_under],
                [first_over, second_under],
                [first_over, second_under],
                [first_under, second_over],
                [first_under, second_over],
                [first_under, second_over],
                [first_under, second_over],
                [first_under, second_over],
                [first_under, second_under],
                [first_under, second_under],
                [first_under, second_under],
                [first_under, second_under],
                [first_under, second_under]]

    return episodes


def get_features(state, action):

    feature_vector = np.zeros(3*7*4)

    for i in range(3):
        for j in range(7):
            for k in range(4):

                if state[0] == i and state[1] == j and action == k:

                    feature_vector[i + 3*j + 3*7*k] = 1

    return feature_vector


def optimise(episodes, policy):

    feature_vector = get_features([0, 6], 3)

    # figure out what goes in here


    return episodes, policy


def main():

    episodes = create_episodes()
    policy = np.random.randint(0, 4, [3, 7])

    margin = 10e99

    while margin > 1:

        margin, weights = optimise(episodes, policy)


main()
