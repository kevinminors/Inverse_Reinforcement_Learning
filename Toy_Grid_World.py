import numpy as np
np.set_printoptions(linewidth=np.nan)

# move small submethods into the methods that call them


def create_real_episodes():

    first_over = [[1, 0], 1, [2, 0], 2, [2, 1], 2, [2, 2], 3, [1, 2], 2, [1, 3], 2]
    second_over = [[1, 4], 1, [2, 4], 2, [2, 5], 2, [2, 6], 3, [1, 6], None]

    first_under = [[1, 0], 3, [0, 0], 2, [0, 1], 2, [0, 2], 3, [1, 2], 2, [1, 3], 2]
    second_under = [[1, 4], 3, [0, 4], 2, [0, 5], 2, [0, 6], 1, [1, 6], None]

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


def generate_random_policy():

    policy = np.array([[2, 1, 2, 2, 2, 1, 1],
                       [3, 1, 3, 0, 3, 1, None],
                       [2, 2, 2, 3, 2, 2, 3]])

    return policy


def create_simulated_episodes(policy):

    current_state = [1, 0]
    current_action = policy[current_state[0], current_state[1]]
    episode = [current_state, current_action]

    while current_action is not None:

        if current_action == 0:

            current_state = [current_state[0], current_state[1] - 1]

        elif current_action == 1:

            current_state = [current_state[0] + 1, current_state[1]]

        elif current_action == 2:

            current_state = [current_state[0], current_state[1] + 1]

        elif current_action == 3:

            current_state = [current_state[0] - 1, current_state[1]]

        current_action = policy[current_state[0], current_state[1]]
        episode.append(current_state)
        episode.append(current_action)

    episodes = [episode for _ in range(20)]

    return episodes


def get_feature_vector_k(k):

    feature_vector = np.zeros(3*7)
    feature_vector[k] = 1

    return feature_vector


def calculate_policy_k_value(episodes, feature_vector, discount):

    value = 0

    for episode in episodes:

        states = episode[::2]

        for k, state in enumerate(states):
            value += feature_vector[state[0] + 3 * state[1]] * discount ** k

    value /= len(episodes)

    return value


def calculate_w_k(k, policies, discount):

    w_k = 0

    feature_vector = get_feature_vector_k(k)
    real_episodes = create_real_episodes()
    perfect_policy_k_value = calculate_policy_k_value(real_episodes, feature_vector, discount)

    for policy in policies:

        simulated_episodes = create_simulated_episodes(policy)
        policy_k_value = calculate_policy_k_value(simulated_episodes, feature_vector, discount)
        w_k += perfect_policy_k_value - policy_k_value

    return w_k


def calculate_value_difference(w_k):
    # value difference = sum w_k * weight_k
    # value_difference = 0
    # value_difference += (perfect_policy_k_value - policy_k_value) * weight_k

    pass


def calculate_gradient(current_weights, policies, discount):

    gradient = []

    for k in range(21):

        w_k = calculate_w_k(k, policies, discount)
        print(w_k)

        # todo start here
        # may need a different loop here
        value_difference = calculate_value_difference(k, w_k, current_weights)


        # if value_difference >= 0:
        #     gradient.append(w_k)
        # else:
        #     gradient.append(-2*w_k)

    return gradient


def main():
    '''
    Method

    Generate value estimate from start state s_0 using episodes aka pi*
    Generate value estimate from start state s_0 using random pi_1

    Calculate term to maximise as a function of weights
    Gradient ascent to find weights that maximise

    Calculate new policy from new weights
    Add new policy to list of policies
    Repeat algorithm
    '''

    learning_rate = 0.01
    max_iterations = 1
    precision = 0.001
    discount = 0.9

    next_weights = np.random.random([3*7])*2 - 1

    random_policy = generate_random_policy()
    policies = [random_policy]

    for _ in range(max_iterations):

        current_weights = next_weights
        gradient = calculate_gradient(current_weights, policies, discount)
        # next_weights = current_weights + learning_rate*gradient
        #
        # reward_function = calculate_reward_function(next_weights)
        # new_policy = maximise_reward_function(reward_function)
        # policies.append(new_policy)
        #
        # step = np.linalg.norm(next_weights - current_weights)
        #
        # if abs(step) <= precision:
        #     break


main()


#
# def get_reward(state, weights):
#
#     feature_vector = get_feature_vector(state)
#     return np.dot(feature_vector, weights)
#
#
#
# def optimise(weights, policy_values):
#
#     pass
#     #
#     # # create array that stores all summed differences
#     # # for each policy and each episode
#     # # find minimum
#     # # use gradient descent to find weights that maximise this minimum
#     # # i.e rewrite optimise in terms of just weights
#     #
#     # for policy in policies:
#     #     for episode in episodes:
#     #         for i in range(len(episode)//2):
#     #
#     #             reward_difference = 0
#     #
#     #             state = episode[2*i]
#     #             action = episode[2*i+1]
#     #
#     #             episode_reward = get_reward(state, weights)
#     #             policy_reward = get_reward(state, weights)
#     #
#     #             reward_difference += episode_reward - policy_reward
#     #
#     #             # calculate difference for all trajectories and all policies
#     #             # implement gradient descent using weights


# episodes_value = calculate_value_function_for_episodes(feature_vectors, real_episodes, discount, weights)
# simulated_value = calculate_value_function_for_episodes(feature_vectors, simulated_episodes, discount, weights)
#
# optimisation_sum = 0
#
# if episodes_value - simulated_value >= 0:
#
#     optimisation_sum += episodes_value - simulated_value
#
# else:
#
#     optimisation_sum -= 2*(episodes_value - simulated_value)
#
# print(optimisation_sum)


# optimise(weights, policy_values)

# do we need to create actual policies?
# then calculate value for policy directly from weights


# episodes = create_episodes()
# policy = np.random.randint(0, 4, [3, 7])
# policies = [policy]
#
# weights = np.random.randint(0, 10, 3 * 7 * 4)
#
# optimise(episodes, policies, weights)

# margin = 10e99
#
# while margin > 1:
#
#     margin, weights = optimise(episodes, policy)

#
# def get_feature_vector(state):
#
#     feature_vector = np.zeros(3*7)
#
#     for i in range(3):
#         for j in range(7):
#
#             if state[0] == i and state[1] == j:
#
#                 feature_vector[i + 3*j] = 1
#
#     return feature_vector

# def generate_all_feature_vectors():
#
#     feature_vectors = []
#
#     for i in range(3):
#         for j in range(7):
#
#             feature_vectors.append(get_feature_vector([i, j]))
#
#     return feature_vectors

# def calculate_value_function_for_episodes(feature_vectors, episodes, discount, weights):
#
#     value_function = np.zeros([len(feature_vectors)])
#
#     for i, feature_vector in enumerate(feature_vectors):
#
#         value = 0
#
#         for j, episode in enumerate(episodes):
#
#             states = episode[::2]
#
#             for k, state in enumerate(states):
#
#                 value += feature_vector[state[0] + 3*state[1]]*discount**k
#
#         value /= len(episodes)
#
#         value_function[i] = value
#
#     return np.dot(value_function, weights)

