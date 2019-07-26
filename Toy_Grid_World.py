import numpy as np
np.set_printoptions(linewidth=np.nan)

# move small submethods into the methods that call them


def create_real_episodes():
    '''
    Create handmade real episodes for IRL method to learn a
    reward function for. The episodes exist in a 3x7 grid world.
    Certain squares are purposely avoided to determine if the
    IRL algorithm can detect these differences
    '''

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
    '''
    A random handmade policy to begin gradient descent method
    '''

    policy = np.array([[2, 1, 2, 2, 2, 1, 1],
                       [3, 1, 3, 0, 3, 1, None],
                       [2, 2, 2, 3, 2, 2, 3]])

    return policy


def create_simulated_episodes(policy):
    '''
    Create list of simulated episodes in GridWorld from the policy
    '''
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
    '''
    Return the feature vector with a 1 in the k-th position
    '''
    feature_vector = np.zeros(3*7)
    feature_vector[k] = 1

    return feature_vector


def calculate_policy_k_value(episodes, feature_vector, discount):
    '''
    Calculate the approximate policy value V^pi(s), taking the average over
    all episodes.

    :param episodes:        list of all episodes to calculate value for
    :param feature_vector:  the function approximator to use as approximation for reward
    :param discount:        amount to decrease value of future reward
    :return:                value V^pi(s) as defined in paper
    '''
    value = 0

    for episode in episodes:

        states = episode[::2]

        for k, state in enumerate(states):
            value += feature_vector[state[0] + 3 * state[1]] * discount ** k

    value /= len(episodes)

    return value


def calculate_w_k(k, policies, discount):
    '''
    Calculate variable v_k

    :param k:           index of the current weight/feature vector
    :param policies:    list of policies to be optimised over
    :param discount:    amount to decrease value of future reward
    :return:            variable for gradient = sum_i V^(pi*)_k - V^(pi_i)_k
    '''

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
    '''
    Calculate the value difference V^(pi*)(s) - V^(pi_1) to determine if
    p(x) = x when x >= 0 or p(x) = -2x when x < 0

    :param w_k:     variable for gradient = sum_i V^(pi*)_k - V^(pi_i)_k
    :return:        value difference
    '''
    # value difference = sum w_k * weight_k
    # value_difference = 0
    # value_difference += (perfect_policy_k_value - policy_k_value) * weight_k

    pass


def calculate_gradient(current_weights, policies, discount):
    '''
    Calculate the gradient to update the current weights in the direction
    of decreasing gradient for gradient descent

    :param current_weights:     the current values for the weights
    :param policies:            the set of policies currently being optimised over
    :param discount:            the amount to decrease value of future rewards
    :return:                    list of gradient components for each weight
    '''

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
    Main

    DESCRIPTION
    Apply inverse reinforcement learning  in https://ai.stanford.edu/~ang/papers/icml00-irl.pdf by Ng
    to https://www.highd-dataset.com/ dataset

    PARAMETERS
     - learning rate:       the step jump for each weight update
     - max_iterations:      the maximum number of times to iterate gradient descent
     - precision:           the difference between consecutive weights to stop descending
     - discount:            amount to decrease confidence in future rewards
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

