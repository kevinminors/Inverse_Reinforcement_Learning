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


def optimise(episodes, policy):



    return episodes, policy


def get_features(state, action):

    # features = np.zeros([2*6*4])
    #
    # for i in range(3):
    #     for j in range(7):
    #         for k in range(5):
    #
    #             if state[0] == i and state[1] == j and action == k:
    #
    #                 features[2*i + 6*j + k] = 1


    # extend feature vector actions as well.
    # double check it is working

    features = np.zeros(3*7)

    for i in range(3):
        for j in range(7):
            if state[0] == i and state[1] == j:

                features[i + 3*j] = 1

    return features


def main():

    episodes = create_episodes()
    policy = np.random.randint(0, 4, [3, 7])

    features = get_features([2, 6], 3)
    print(features)

    # margin = 10e99
    #
    # while margin > 1:
    #
    #     margin, weights = optimise(episodes, policy)


main()
