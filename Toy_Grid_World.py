import numpy as np


def create_episodes():

    first_over = [[1, 0], 1, [2, 0], 2, [2, 1], 2, [2, 2], 3, [1, 2], 2, [1, 3], 2]
    second_over = [[1, 4], 1, [2, 4], 2, [2, 5], 2, [2, 6], 3]

    first_under = [[1, 0], 3, [0, 0], 2, [0, 1], 2, [0, 2], 3, [1, 2], 2, [1, 3], 2]
    second_under = [[1, 4], 3, [0, 4], 2, [0, 5], 2, [0, 6], 1]

    episode1 = first_over
    episode1.extend(second_over)

    episode2 = first_over
    episode2.extend(second_under)

    episode3 = first_under
    episode3.extend(second_over)

    episode4 = first_under
    episode4.extend(second_under)

    episodes = [episode1,
                episode1,
                episode1,
                episode1,
                episode1,
                episode2,
                episode2,
                episode2,
                episode2,
                episode2,
                episode3,
                episode3,
                episode3,
                episode3,
                episode3,
                episode4,
                episode4,
                episode4,
                episode4,
                episode4]

    return episodes


def get_features(state, action):

    feature_vector = np.zeros(3*7*4)

    for i in range(3):
        for j in range(7):
            for k in range(4):

                if state[0] == i and state[1] == j and action == k:

                    feature_vector[i + 3*j + 3*7*k] = 1

    return feature_vector


def get_reward(state, action, weights):

    feature_vector = get_features(state, action)
    return np.dot(feature_vector, weights)


def optimise(episodes, policies, weights):

    # create array that stores all summed differences
    # for each policy and each episode
    # find minimum
    # use gradient descent to find weights that maximise this minimum
    # i.e rewrite optimise in terms of just weights

    for policy in policies:
        for episode in episodes:
            for i in range(len(episode)//2):

                reward_difference = 0

                state = episode[2*i]
                action = episode[2*i+1]

                episode_reward = get_reward(state, action, weights)
                policy_reward = get_reward(state, policy[state[0], state[1]], weights)

                reward_difference += episode_reward - policy_reward

                # calculate difference for all trajectories and all policies
                # implement gradient descent using weights


def main():

    episodes = create_episodes()
    policy = np.random.randint(0, 4, [3, 7])
    policies = [policy]

    weights = np.random.randint(0, 10, 3 * 7 * 4)

    optimise(episodes, policies, weights)

    # margin = 10e99
    #
    # while margin > 1:
    #
    #     margin, weights = optimise(episodes, policy)


main()
