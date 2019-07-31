import numpy as np
from collections import Counter
np.set_printoptions(linewidth=np.nan)


def convert_state_to_vector_index(state):
    """
    Helper function to convert state in Gridworld
    to an index used for weights and feature vectors.

    :param state: [x, y] location in Gridworld
    :return: corresponding index for feature/weights vectors
    """
    return 7 * (2 - state[0]) + state[1]


def convert_vector_index_to_state(index):
    """
    Helper function to convert vector index used
    in feature/weight vectors into a state in Gridworld.

    :param index: index used in feature/weight vectors
    :return: corresponding [x, y] state in Gridworld
    """
    state = [0, 0]

    state[1] = index % 7
    state[0] = 2 - (index - state[1])//7

    return state


def calculate_gradient(weights, policies):
    """
    Calculate the direction of the gradient to update the weights
    in order to find the weights that maximise the sum of the
    differences between the real episode values and random policy values.

    In particular, the sum we want to find weights for is:

    sum_{i=1}^k p( V^{pi*}(s) - V^{pi_i}(s) )

    where abs(alpha_i) <= 1 for all weights alpha_i
    and p(x) = x if x >= 0 and p(x) = 2x if x < 0, which
    is a penalty term for when V^{pi*}(s) < V^{pi_i}(s).

    Full details can be found in section 5 of
    'Algorithms for Inverse Reinforcement Learning' by Ng and Russel
    https://ai.stanford.edu/~ang/papers/icml00-irl.pdf.

    :param weights:             the current values for the weights
    :param policies:            the set of policies currently being optimised over
    :return:                    list of gradient components for each weight
    """

    def get_feature_vector(feature_vector_number):
        """
        Helper function to get feature vector that is zero everywhere
        except a 1 in the given index

        :param feature_vector_number:   index for the 1 in the vector
        :return:                        feature vector with 1 in that index
        """
        feature_vector = np.zeros(3 * 7)
        feature_vector[feature_vector_number] = 1

        return feature_vector

    def create_simulated_episodes(current_policy):
        """
        Create simulated episode in GridWorld from the given policy

        :param current_policy:  policy to use to create episode
        :return:                episode created using given policy
        """
        current_state = [1, 0]
        current_action = current_policy[current_state[0], current_state[1]]
        episode = [current_state, current_action]

        while current_state != [1, 6]:

            if current_action == 0:

                current_state = [current_state[0], current_state[1] - 1]

            elif current_action == 1:

                current_state = [current_state[0] + 1, current_state[1]]

            elif current_action == 2:

                current_state = [current_state[0], current_state[1] + 1]

            elif current_action == 3:

                current_state = [current_state[0] - 1, current_state[1]]

            current_action = current_policy[current_state[0], current_state[1]]
            episode.append(current_state)
            episode.append(current_action)

        episodes = [episode]

        return episodes

    def create_real_episodes():
        """
        Create real episodes for the inverse reinforcement
        learning to learn from. The episodes exist in a 3x7
        Gridworld. They begin in [1,0] and end in [1,6]. All
        four episodes avoid locations [1, 1], [0, 3], [2, 3],
        and [1, 5]. The goal is for this IRL method to reproduce
        the fact that certain squares are avoided.

        :return: episodes for IRL method to learn from
        """

        first_over_second_over = [[1, 0], 1, [2, 0], 2, [2, 1], 2, [2, 2], 3, [1, 2], 2, [1, 3], 2,
                                  [1, 4], 1, [2, 4], 2, [2, 5], 2, [2, 6], 3, [1, 6], None]

        first_over_second_under = [[1, 0], 1, [2, 0], 2, [2, 1], 2, [2, 2], 3, [1, 2], 2, [1, 3], 2,
                                   [1, 4], 3, [0, 4], 2, [0, 5], 2, [0, 6], 1, [1, 6], None]

        first_under_second_over = [[1, 0], 3, [0, 0], 2, [0, 1], 2, [0, 2], 3, [1, 2], 2, [1, 3], 2,
                                   [1, 4], 1, [2, 4], 2, [2, 5], 2, [2, 6], 3, [1, 6], None]

        first_under_second_under = [[1, 0], 3, [0, 0], 2, [0, 1], 2, [0, 2], 3, [1, 2], 2, [1, 3], 2,
                                    [1, 4], 3, [0, 4], 2, [0, 5], 2, [0, 6], 1, [1, 6], None]

        episodes = [first_over_second_over,
                    first_over_second_under,
                    first_under_second_over,
                    first_under_second_under]

        return episodes

    def calculate_approximate_reward(episodes, feature_vector):
        """
        Calculate the average empirical return V_i^{pi} for
        feature vector i and policy pi derived from the given episodes.
        V_i^{pi} is defined as:

        V^{pi_i}(s) = sum_{j=0}^{infinity} feature_vector_i(state(j)) * discount^j

        where the sum increments over states in the episode and
        the power of the discount term.

        We calculate this value for each episode and then take the average
        over all episodes.

        Full details can be found in section 5 of
        'Algorithms for Inverse Reinforcement Learning' by Ng and Russel
        https://ai.stanford.edu/~ang/papers/icml00-irl.pdf.

        :param episodes:        list of episodes to grab states from
        :param feature_vector:  vector to use as approximation for reward function
        :return:                mean value of V_i^{pi} for all episodes
        """
        episode_values = np.zeros(len(episodes))

        for i, episode in enumerate(episodes):

            # todo rewrite current_ep and episode values using list comprehension

            current_episode_value = 0
            states = episode[::2]

            for k, state in enumerate(states):
                current_episode_value += feature_vector[convert_state_to_vector_index(state)] * REWARD_DISCOUNT_FACTOR ** k

            episode_values[i] = current_episode_value

        mean_episode_value = np.mean(episode_values)

        return mean_episode_value

    def calculate_p_gradient(current_policy):
        """
        Taking the partial derivative of the sum:

        sum_{i=1}^k p( V^{pi*}(s) - V^{pi_i}(s) )

        with respect to each weight alpha_n results
        in the following sum:

        sum_{i=1}^k p'( V^{pi*}(s) - V^{pi_i}(s) )
                        * ( V_n^{pi*}(s) - V_n^{pi_i}(s) )

        This method calculates p'( V^{pi*}(s) - V^{pi_i}(s) ).

        :param current_policy:      Policy used to compare against expert episodes
        :return:                    Value of p' in gradient calculation
        """
        sum_product = np.zeros(21)

        real_episodes = create_real_episodes()

        for feature_vector_number in range(21):

            feature_vector = get_feature_vector(feature_vector_number)
            perfect_policy_weight_value = calculate_approximate_reward(real_episodes, feature_vector)

            simulated_episodes = create_simulated_episodes(current_policy)
            policy_weight_value = calculate_approximate_reward(simulated_episodes, feature_vector)

            sum_product[feature_vector_number] = weights[feature_vector_number] * (perfect_policy_weight_value
                                                                                   - policy_weight_value)

        weight_sum = sum(sum_product)

        if weight_sum >= 0:
            return P_FUNCTION_REWARD
        else:
            return P_FUNCTION_PENALTY

    def calculate_value_function_difference(feature_vector_number, current_policy):
        """
        Taking the partial derivative of the sum:

        sum_{i=1}^k p( V^{pi*}(s) - V^{pi_i}(s) )

        with respect to each weight alpha_n results
        in the following sum:

        sum_{i=1}^k p'( V^{pi*}(s) - V^{pi_i}(s) )
                        * ( V_n^{pi*}(s) - V_n^{pi_i}(s) )

        This method calculates V_n^{pi*}(s) - V_n^{pi_i}(s).

        :param feature_vector_number:   Index to use to generate feature vector
        :param current_policy:          Policy used to compare against expert episodes
        :return:                        Value of difference between policy values
        """
        feature_vector = get_feature_vector(feature_vector_number)
        real_episodes = create_real_episodes()
        perfect_policy_weight_value = calculate_approximate_reward(real_episodes, feature_vector)

        simulated_episodes = create_simulated_episodes(current_policy)
        policy_weight_value = calculate_approximate_reward(simulated_episodes, feature_vector)

        return perfect_policy_weight_value - policy_weight_value

    # todo use list comprehensions here

    gradient = []

    for j in range(21):

        weight_gradient = 0

        for policy in policies:

            p_gradient = calculate_p_gradient(policy)
            value_function_difference = calculate_value_function_difference(j, policy)

            weight_gradient += p_gradient*value_function_difference

        gradient.append(weight_gradient)

    return gradient


def step_model(state, action, weights, previous_state):
    """
    Step through the model given a state and an action.
    Return the new state, immediate reward, and whether or
    not it is a terminal state.

    If the agent moves off Gridworld or back to the previous
    state, there is a negative reward.

    Sub methods filter through different values for each state
    element and then consider the appropriate action and reward.

    Action = 0 (move left)
    Action = 1 (move up)
    Action = 2 (move right)
    Action = 3 (move down)

    :param state:           Current state of the agent
    :param action:          Current action the agent takes
    :param weights:         Weights that approximate rewards for various actions
    :param previous_state:  Previous state of the agent
    :return:                New state, immediate reward, and if new state is a terminal state
    """
    def step_state_zero():

        # todo could move index and previous state check to top

        def step_state_zero_zero():

            index = convert_state_to_vector_index(state)

            if action == 0:

                return state, EXIT_BOUNDARY_REWARD, False

            elif action == 1:

                new_state = [state[0] + 1, state[1]]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

            elif action == 2:

                new_state = [state[0], state[1] + 1]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

            elif action == 3:

                return state, EXIT_BOUNDARY_REWARD, False

        def step_state_zero_six():

            index = convert_state_to_vector_index(state)

            if action == 0:

                new_state = [state[0], state[1] - 1]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

            elif action == 1:

                return [state[0] + 1, state[1]], TERMINAL_STATE_REWARD, True

            elif action == 2:

                return state, EXIT_BOUNDARY_REWARD, False

            elif action == 3:

                return state, EXIT_BOUNDARY_REWARD, False

        def step_state_zero_middle():

            index = convert_state_to_vector_index(state)

            if action == 0:

                new_state = [state[0], state[1] - 1]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

            elif action == 1:

                new_state = [state[0] + 1, state[1]]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

            elif action == 2:

                new_state = [state[0], state[1] + 1]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

            elif action == 3:

                return state, EXIT_BOUNDARY_REWARD, False

        if state[1] == 0:

            return step_state_zero_zero()

        elif state[1] == 6:

            return step_state_zero_six()

        else:

            return step_state_zero_middle()

    def step_state_one():

        def step_state_one_zero():

            index = convert_state_to_vector_index(state)

            if action == 0:

                return state, EXIT_BOUNDARY_REWARD, False

            elif action == 1:

                new_state = [state[0] + 1, state[1]]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

            elif action == 2:

                new_state = [state[0], state[1] + 1]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

            elif action == 3:

                new_state = [state[0] - 1, state[1]]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

        def step_state_one_five():

            index = convert_state_to_vector_index(state)

            if action == 0:

                new_state = [state[0], state[1] - 1]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

            elif action == 1:

                new_state = [state[0] + 1, state[1]]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

            elif action == 2:

                return [state[0], state[1] + 1], TERMINAL_STATE_REWARD, True

            elif action == 3:

                new_state = [state[0] - 1, state[1]]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

        def step_state_one_middle():

            index = convert_state_to_vector_index(state)

            if action == 0:

                new_state = [state[0], state[1] - 1]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

            elif action == 1:

                new_state = [state[0] + 1, state[1]]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

            elif action == 2:

                new_state = [state[0], state[1] + 1]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

            elif action == 3:

                new_state = [state[0] - 1, state[1]]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

        if state[1] == 0:

            return step_state_one_zero()

        elif state[1] == 5:

            return step_state_one_five()

        else:

            return step_state_one_middle()

    def step_state_two():

        def step_state_two_zero():

            index = convert_state_to_vector_index(state)

            if action == 0:

                return state, EXIT_BOUNDARY_REWARD, False

            elif action == 1:

                return state, EXIT_BOUNDARY_REWARD, False

            elif action == 2:

                new_state = [state[0], state[1] + 1]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

            elif action == 3:

                new_state = [state[0] - 1, state[1]]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

        def step_state_two_six():

            index = convert_state_to_vector_index(state)

            if action == 0:

                new_state = [state[0], state[1] - 1]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

            elif action == 1:

                return state, EXIT_BOUNDARY_REWARD, False

            elif action == 2:

                return state, EXIT_BOUNDARY_REWARD, False

            elif action == 3:

                return [state[0] - 1, state[1]], TERMINAL_STATE_REWARD, True

        def step_state_two_middle():

            index = convert_state_to_vector_index(state)

            if action == 0:

                new_state = [state[0], state[1] - 1]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

            elif action == 1:

                return state, EXIT_BOUNDARY_REWARD, False

            elif action == 2:

                new_state = [state[0], state[1] + 1]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

            elif action == 3:

                new_state = [state[0] - 1, state[1]]

                if new_state != previous_state:
                    return new_state, weights[index], False
                else:
                    return new_state, REPEAT_PREVIOUS_STATE_REWARD, False

        if state[1] == 0:

            return step_state_two_zero()

        elif state[1] == 6:

            return step_state_two_six()

        else:

            return step_state_two_middle()

    if state[0] == 0:

        return step_state_zero()

    elif state[0] == 1:

        return step_state_one()

    elif state[0] == 2:

        return step_state_two()


def calculate_maximal_reward_policy(weights):
    """

    :param weights:
    :return:
    """
    policy = np.random.randint(0, 4, [3, 7])
    action_value_function = np.zeros([3, 7, 4])
    state_action_counter = np.zeros([3, 7, 4])
    state_counter = np.zeros([3, 7])

    for i in range(MAXIMUM_NUMBER_OF_POLICY_EPISODES):

        eligibility_traces = np.zeros([3, 7, 4])

        state = [1, 0]
        previous_state = [0, 6]
        action = policy[state[0], state[1]]

        episode = list()
        episode.append(state)
        episode.append(action)

        # todo there has to be a better way to do the following logic

        terminal_state = False

        while not terminal_state:

            new_state, reward, terminal_state = step_model(state, action, weights, previous_state)

            if terminal_state:
                break

            state_counter[state[0], state[1]] += 1

            state_action_pair = state[0], state[1], action
            state_action_counter[state_action_pair] += 1

            epsilon = EPSILON_RATIO_VALUE / (state_counter[state[0], state[1]] + EPSILON_RATIO_VALUE)
            probability = np.random.rand()

            if probability <= epsilon:
                policy[new_state[0], new_state[1]] = np.random.randint(0, 4)

            else:
                policy[new_state[0], new_state[1]] = np.argmax(action_value_function[new_state[0], new_state[1], :])

            new_action = policy[new_state[0], new_state[1]]
            episode.append(new_state)
            episode.append(new_action)
            delta = (reward + action_value_function[new_state[0], new_state[1], new_action]
                     - action_value_function[state_action_pair])

            eligibility_traces[state_action_pair] += 1

            action_value_function[state_action_pair] += (delta * eligibility_traces[state_action_pair]
                                                         / state_action_counter[state_action_pair])
            eligibility_traces[state_action_pair] *= LAMBDA * REWARD_DISCOUNT_FACTOR

            previous_state = state
            state = new_state
            action = new_action

    return policy


def main():
    """
    Main

    DESCRIPTION
    Apply inverse reinforcement learning  in https://ai.stanford.edu/~ang/papers/icml00-irl.pdf by Ng
    to https://www.highd-dataset.com/ dataset

    PARAMETERS
     - learning rate:       the step jump for each weight update
     - max_iterations:      the maximum number of times to iterate gradient descent
     - precision:           the difference between consecutive weights to stop descending
     - discount:            amount to decrease confidence in future rewards
    """

    def generate_random_policy():
        """
        A random handmade policy to begin gradient descent method
        """
        policy = np.array([[2, 2, 2, 2, 2, 2, 2],
                           [2, 2, 2, 2, 2, 2, 2],
                           [2, 2, 2, 2, 2, 2, 2]])

        return policy

    random_policy = generate_random_policy()
    policies = [random_policy]
    rewards = []

    while len(policies) < MAXIMUM_NUMBER_OF_POLICIES:

        print('Progress:', len(policies) / MAXIMUM_NUMBER_OF_POLICIES * 100, '%')

        next_weights = np.zeros([3*7])
        current_weights = np.zeros([3*7])
        policy_appended = False

        for i in range(MAXIMUM_WEIGHT_UPDATES):

            current_weights = next_weights
            gradients = calculate_gradient(current_weights, policies)
            weights_change = [LEARNING_RATE * gradient for gradient in gradients]
            next_weights = current_weights + weights_change

            next_weights = np.minimum(next_weights, np.ones(len(next_weights)))
            next_weights = np.maximum(next_weights, -1*np.ones(len(next_weights)))

            step = np.linalg.norm(next_weights - current_weights)

            if step <= REQUIRED_STEP_PRECISION:
                new_policy = calculate_maximal_reward_policy(next_weights)
                rewards.append(str(np.resize(next_weights, [3, 7])))
                policies.append(new_policy)
                policy_appended = True
                break

        if not policy_appended:

            probability = np.random.rand()

            if probability >= 0.5:

                new_policy = calculate_maximal_reward_policy(next_weights)
                rewards.append(str(np.resize(next_weights, [3, 7])))

            else:

                new_policy = calculate_maximal_reward_policy(current_weights)
                rewards.append(str(np.resize(current_weights, [3, 7])))

            policies.append(new_policy)

    print(Counter(rewards).most_common(4))


LEARNING_RATE = 0.1
MAXIMUM_WEIGHT_UPDATES = 50
MAXIMUM_NUMBER_OF_POLICIES = 3
REQUIRED_STEP_PRECISION = 0.001

REWARD_DISCOUNT_FACTOR = 1
EXIT_BOUNDARY_REWARD = -1
REPEAT_PREVIOUS_STATE_REWARD = -1  # do we really need this?
TERMINAL_STATE_REWARD = 1

P_FUNCTION_REWARD = 1
P_FUNCTION_PENALTY = 2  # 10000

MAXIMUM_NUMBER_OF_POLICY_EPISODES = 10000
LAMBDA = 0.9
EPSILON_RATIO_VALUE = 1000  # should be less than half of num_of_policy_episodes

# todo remove all hardcoded numbers
# todo make parameters needed everywhere global
# todo use list comprehensions to make code faster
# todo try IRL with one unique real episode
# todo figure out good way to report final rewards

main()
