import numpy as np
np.set_printoptions(linewidth=np.nan)

# move small submethods into the methods that call them
# remove all hardcoded numbers
# replace i, j, k by feature_vector_number etc
'''
move from using 

value = 0
loop
    value += something

to using 

loop
value[j] = something

to using list comphrehension and then summing the list
'''


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


def generate_random_policy():
    '''
    A random handmade policy to begin gradient descent method
    '''

    policy = np.array([[2, 1, 2, 2, 2, 1, 1],
                       [3, 1, 3, 0, 3, 1, None],
                       [2, 2, 2, 3, 2, 2, 3]])

    return policy


def get_feature_vector(feature_vector_number):
    '''
    Return the feature vector with a 1 in the k-th position
    '''
    feature_vector = np.zeros(3*7)
    feature_vector[feature_vector_number] = 1

    return feature_vector


def calculate_approximate_reward(episodes, feature_vector, discount):
    '''
    Calculate the approximate policy value V^pi(s), taking the average over
    all episodes.

    :param episodes:        list of all episodes to calculate value for
    :param feature_vector:  the function approximator to use as approximation for reward
    :param discount:        amount to decrease value of future reward
    :return:                value V^pi(s) as defined in paper
    '''
    episode_values = np.zeros(len(episodes))

    for i, episode in enumerate(episodes):

        current_episode_value = 0
        states = episode[::2]

        for k, state in enumerate(states):
            current_episode_value += feature_vector[state[0] + 3 * state[1]] * discount ** k

        episode_values[i] = current_episode_value

    mean_episode_value = np.mean(episode_values)

    return mean_episode_value


def calculate_p_gradient(weights, policy, discount):

    sum_product = np.zeros(21)

    real_episodes = create_real_episodes()

    for feature_vector_number in range(21):
        feature_vector = get_feature_vector(feature_vector_number)
        perfect_policy_weight_value = calculate_approximate_reward(real_episodes, feature_vector, discount)

        simulated_episodes = create_simulated_episodes(policy)
        policy_weight_value = calculate_approximate_reward(simulated_episodes, feature_vector, discount)

        sum_product[feature_vector_number] = weights[feature_vector_number]*(perfect_policy_weight_value
                                                                                     - policy_weight_value)

    weight_sum = sum(sum_product)

    if weight_sum >= 0:
        return 1
    else:
        return 2


def calculate_value_function_difference(feature_vector_number, policy, discount):

    feature_vector = get_feature_vector(feature_vector_number)
    real_episodes = create_real_episodes()
    perfect_policy_weight_value = calculate_approximate_reward(real_episodes, feature_vector, discount)

    simulated_episodes = create_simulated_episodes(policy)
    policy_weight_value = calculate_approximate_reward(simulated_episodes, feature_vector, discount)

    return perfect_policy_weight_value - policy_weight_value


def calculate_gradient(weights, policies, discount):
    '''
    Calculate the gradient to update the current weights in the direction
    of decreasing gradient for gradient descent

    :param weights:     the current values for the weights
    :param policies:            the set of policies currently being optimised over
    :param discount:            the amount to decrease value of future rewards
    :return:                    list of gradient components for each weight
    '''

    gradient = []

    for j in range(21):

        weight_gradient = 0

        for policy in policies:

            p_gradient = calculate_p_gradient(weights, policy, discount)
            value_function_difference = calculate_value_function_difference(j, policy, discount)

            weight_gradient += p_gradient*value_function_difference

        gradient.append(weight_gradient)

    return gradient


def calculate_maximal_reward_policy(weights):

    # todo start here
    # may need some serious RL here
    # to calculate optimal policy from reward function

    return 0


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
    learning_rate = 0.1
    max_weight_updates = 1000000
    max_number_of_policies = 100
    precision = 0.000000001
    discount = 0.9

    next_weights = np.random.random([3*7])*2 - 1

    random_policy = generate_random_policy()
    policies = [random_policy]

    while len(policies) < max_number_of_policies:

        for i in range(max_weight_updates):

            current_weights = next_weights
            gradients = calculate_gradient(current_weights, policies, discount)
            weights_change = [learning_rate*gradient for gradient in gradients]
            next_weights = current_weights - weights_change

            next_weights = np.minimum(next_weights, np.ones(len(next_weights)))
            next_weights = np.maximum(next_weights, -1*np.ones(len(next_weights)))

            step = np.linalg.norm(next_weights - current_weights)

            if abs(step) <= precision:
                new_policy = calculate_maximal_reward_policy(next_weights)
                policies.append(new_policy)
                next_weights = np.random.random([3 * 7]) * 2 - 1
                break

    print(np.resize(next_weights, [3, 7]))


main()
