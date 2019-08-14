import numpy as np
np.set_printoptions(linewidth=np.nan)

LEARNING_RATE = 0.1
MAXIMUM_WEIGHT_UPDATES = 50
MAXIMUM_NUMBER_OF_POLICIES = 10
REQUIRED_STEP_PRECISION = 0.001

REWARD_DISCOUNT_FACTOR = 1
EXIT_BOUNDARY_REWARD = -1
REPEAT_PREVIOUS_STATE_REWARD = -1
TERMINAL_STATE_REWARD = 1

P_FUNCTION_REWARD = 1
P_FUNCTION_PENALTY = 2

MAXIMUM_NUMBER_OF_POLICY_EPISODES = 10000
MAXIMUM_POLICY_ITERATIONS = 1000
LAMBDA = 0.9
EPSILON_RATIO_VALUE = 1000

AVOID_COLLISION_REWARD = 0
COLLISION_PENALTY = 0
TOO_SLOW_PENALTY = 0

RANDOM_STATE_CHANGE_PROBABILITY = 0.7

# todo add print statements everywhere to see what is actually going on

def main():
    """
    We are demonstrating the use of inverse reinforcement learning in
    calculating a reward function that would reproduce observed behaviour.
    This behaviour is assumed to be expert behaviour. The goal is to learn
    a reward function that would allow a reinforcement agent to reproduce
    the observed behaviour. The observed behaviour takes the form of trajectories
    (sequences of states and actions). We find the reward function by using gradient descent
    to find the weights that maximise the difference between the observed behaviour and
    any other policy behaviour. Given a new set of weights, we calculate a new policy
    to consider using the SARSA(lambda) method.

    Full details can be found in section 5 of
    'Algorithms for Inverse Reinforcement Learning' by Ng and Russel
    https://ai.stanford.edu/~ang/papers/icml00-irl.pdf.

    Details of gradient descent can be found here:
    https://en.wikipedia.org/wiki/Gradient_descent

    Details of the SARSA(lambda) method can be found here:
    http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf
    """

    def calculate_gradient(weights, current_policies):
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
        :param current_policies:            the set of policies currently being optimised over
        :return:                    list of gradient components for each weight
        """

        def get_feature_vector(feature_vector_number):
            """
            Helper function to get feature vector that is zero everywhere
            except a 1 in the given index

            :param feature_vector_number:   index for the 1 in the vector
            :return:                        feature vector with 1 in that index
            """
            feature_vector = np.zeros(state_bin_length)
            feature_vector[feature_vector_number] = 1

            return feature_vector

        def create_simulated_episodes(current_policy):
            """
            Create simulated episode in GridWorld from the given policy

            :param current_policy:  policy to use to create episode
            :return:                episode created using given policy
            """
            current_state = np.random.randint(0, state_bin_length)
            current_action = current_policy[current_state]
            episode = [current_state, current_action]
            max_episode_length = generate_max_episode_length()

            while (current_state != 0) and (current_state != state_bin_length - 1) and (len(episode) < max_episode_length):
                # print()
                # print('current state')
                # print(current_state)
                # print()
                # print('episode')
                # print(episode, len(episode))
                # print()
                # print('current action')
                # print(current_action)

                # todo add in randomness here. Ttc doesn't always change according to these actions.
                # Sometimes its the opposite!

                if current_action == 0:

                    current_state += 1

                elif current_action == 1:

                    random_number = np.random.random()

                    if random_number < RANDOM_STATE_CHANGE_PROBABILITY:
                        change = np.random.choice([-1, 1])
                        current_state += change

                elif current_action == 2:

                    current_state -= 1

                current_action = current_policy[current_state]
                episode.append(current_state)
                episode.append(current_action)

            episodes = [episode]

            return episodes

        def calculate_reward(episodes, feature_vector):
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

            for n, episode in enumerate(episodes):
                states = episode[::2]
                current_episode_value = [feature_vector[state] * REWARD_DISCOUNT_FACTOR ** k
                                         for k, state in enumerate(states)]

                episode_values[n] = sum(current_episode_value)
                # print()
                # print('states')
                # print(states)
                # print()
                # print('current episode value')
                # print(current_episode_value)

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
            sum_product = np.zeros(state_bin_length)

            for feature_vector_number in range(state_bin_length):
                feature_vector = get_feature_vector(feature_vector_number)
                perfect_policy_weight_value = calculate_reward(real_episodes, feature_vector)

                simulated_episodes = create_simulated_episodes(current_policy)
                policy_weight_value = calculate_reward(simulated_episodes, feature_vector)

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
            perfect_policy_weight_value = calculate_reward(real_episodes, feature_vector)

            simulated_episodes = create_simulated_episodes(current_policy)
            policy_weight_value = calculate_reward(simulated_episodes, feature_vector)

            return perfect_policy_weight_value - policy_weight_value

        gradient = np.zeros(state_bin_length)

        for j in range(state_bin_length):
            # print('gradient entry', j, 'being calculated')
            weight_gradient = [calculate_p_gradient(policy) * calculate_value_function_difference(j, policy)
                               for policy in current_policies]
            gradient[j] = (sum(weight_gradient))

        return gradient

    def step_model(state, action, weights):
        """
        Step through the model given a state and an action.
        Return the new state, immediate reward, and whether or
        not it is a terminal state.

        If the agent moves off Gridworld or back to the previous
        state, there is a negative reward.

        Sub methods filter through different values for each state
        element and then consider the appropriate action and reward.

        Action = 0 (slow down)
        Action = 1 (maintain speed)
        Action = 2 (speed up)

        :param state:           Current state of the agent
        :param action:          Current action the agent takes
        :param weights:         Weights that approximate rewards for various actions
        :return:                New state, immediate reward, and if new state is a terminal state
        """

        def step_state_middle():

            if action == 0:

                return state + 1, weights[state], False

            elif action == 1:

                return state, weights[state], False

            elif action == 2:

                return state - 1, weights[state], False

        def step_state_zero():

            if action == 0:

                return state + 1, AVOID_COLLISION_REWARD, False

            elif action == 1:

                return state, weights[state], False

            elif action == 2:

                return state, COLLISION_PENALTY, True

        def step_state_max():

            if action == 0:

                return state, TOO_SLOW_PENALTY, True

            elif action == 1:

                return state, weights[state], False

            elif action == 2:

                return state - 1, weights[state], False

        if (state > 0) and (state < state_bin_length - 1):

            return step_state_middle()

        elif state == 0:

            return step_state_zero()

        elif state == state_bin_length - 1:

            return step_state_max()

    def calculate_policy(weights):
        """
        The weights that we calculate can also be
        interpreted as rewards. For the given set of weights,
        we calculate a policy that receives a maximal total reward
        through an episode of following the policy.

        We calculate this policy using the SARSA(lambda) method
        found in these lecture notes:

        http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf

        :param weights:     approximate values for rewards used to calculate policy
        :return:            policy that maps states to actions in a way that maximises
                            the total reward received
        """
        policy = np.random.randint(0, 3, state_bin_length)
        action_value_function = np.zeros([state_bin_length, 3])
        state_action_counter = np.zeros([state_bin_length, 3])
        state_counter = np.zeros(state_bin_length)

        # todo what does this policy mean???

        # print('working on policy')
        for m in range(MAXIMUM_NUMBER_OF_POLICY_EPISODES):

            eligibility_traces = np.zeros([state_bin_length, 3])

            state = np.random.randint(0, state_bin_length)
            action = policy[state]

            new_state, reward, terminal_state = step_model(state, action, weights)
            counter = 0

            # todo add maximum length for policy here. May never reach a terminal state
            while (not terminal_state) and (counter < MAXIMUM_POLICY_ITERATIONS):
                # print('current policy counter', counter)
                counter += 1
                state_counter[state] += 1

                state_action_pair = state, action
                state_action_counter[state_action_pair] += 1

                epsilon = EPSILON_RATIO_VALUE / (state_counter[state] + EPSILON_RATIO_VALUE)
                explore_probability = np.random.rand()

                if explore_probability <= epsilon:

                    policy[new_state] = np.random.randint(0, 3)

                else:

                    policy[new_state] = np.argmax(action_value_function[new_state, :])

                new_action = policy[new_state]
                delta = (reward + action_value_function[new_state, new_action]
                         - action_value_function[state_action_pair])

                eligibility_traces[state_action_pair] += 1

                action_value_function[state_action_pair] += (delta * eligibility_traces[state_action_pair]
                                                             / state_action_counter[state_action_pair])
                eligibility_traces[state_action_pair] *= LAMBDA * REWARD_DISCOUNT_FACTOR

                state = new_state
                action = new_action

                new_state, reward, terminal_state = step_model(state, action, weights)

        print()
        print('policy calculated')
        print(policy)

        return policy

    def generate_initial_policy(number_of_states):
        """
        A simple policy used to begin the inverse reinforcement learning
        method. When we are calculating the weights using gradient
        descent, we want to maximise the value function difference
        between the observed behaviour and any other policy. This random
        policy is the first policy we begin with.

        Action 1 = move right

        :return:    policy filled with all 2s
        """
        # policy = np.array([[1, 1, 1, 1, 1, 1, 1],
        #                    [1, 1, 1, 1, 1, 1, 1],
        #                    [1, 1, 1, 1, 1, 1, 1]])

        policy = np.random.randint(0, 3, number_of_states)

        return policy

    def create_real_episodes():
        """
        Create real episodes for the inverse reinforcement
        learning to learn from. The episodes exist in a 3x7
        Gridworld. They begin in [1,0] and end in [1,6]. Both
        episodes avoid the squares in the middle of Gridworld.
        These squares are [1, 1], [1, 2], [1, 3], [1, 4], and
        [1, 5]. The goal is for this IRL method to reproduce
        the fact that certain squares are avoided.

        Action = 0 (move down)
        Action = 1 (move right)
        Action = 2 (move up)

        :return: episodes for IRL method to learn from
        """
        # all_over = [[1, 0], 0, [2, 0], 1, [2, 1], 1, [2, 2], 1, [2, 3], 1,
        #             [2, 4], 1, [2, 5], 1, [2, 6], 2, [1, 6], None]
        #
        # all_under = [[1, 0], 2, [0, 0], 1, [0, 1], 1, [0, 2], 1, [0, 3], 1,
        #              [0, 4], 1, [0, 5], 1, [0, 6], 0, [1, 6], None]
        #
        # episodes = [all_over,
        #             all_under]

        trajectories = list()
        trajectory_lengths = list()

        with open('trajectories.txt', 'r') as g:
            for line in g:
                trajectories.append(eval(line))
                trajectory_lengths.append(len(eval(line)))

        return trajectories, trajectory_lengths

    def generate_max_episode_length():

        return np.random.randint(min(real_episode_lengths), max(real_episode_lengths) + 1)

    with open('details.txt', 'r') as f:
        details = f.readlines()

    state_bin_maximum = int(details[0])
    state_bin_step_size = int(details[1])
    state_bin_length = len(range(0, state_bin_maximum + state_bin_step_size, state_bin_step_size))

    real_episodes, real_episode_lengths = create_real_episodes()
    real_episodes = [real_episodes[5]]
    real_episode_lengths = [real_episode_lengths[5]]

    print()
    print('real episode')
    print(real_episodes[0])

    # import matplotlib.pyplot as plt
    # for episode in real_episodes[:100]:
    #     states = episode[::2]
    #     plt.plot(states)
    # plt.ylabel('State')
    # plt.xlabel('Index in Trajectory')
    # plt.title('Discrete State Trajectories (first 100)')
    # plt.show()

    random_policy = generate_initial_policy(state_bin_length)
    policies = [random_policy]
    rewards = []
    final_reward = np.zeros(state_bin_length)

    while len(policies) - 1 < MAXIMUM_NUMBER_OF_POLICIES:

        # print('Progress:', len(policies) / MAXIMUM_NUMBER_OF_POLICIES * 100, '%')

        next_weights = np.zeros(state_bin_length)
        current_weights = np.zeros(state_bin_length)
        policy_appended = False

        for i in range(MAXIMUM_WEIGHT_UPDATES):

            # print('number of weight updates', i)
            current_weights = next_weights
            gradients = calculate_gradient(current_weights, policies)
            weights_change = [LEARNING_RATE * gradient for gradient in gradients]
            next_weights = current_weights + weights_change

            next_weights = np.minimum(next_weights, np.ones(len(next_weights)))
            next_weights = np.maximum(next_weights, -1*np.ones(len(next_weights)))

            step = np.linalg.norm(next_weights - current_weights)
            # print('step', step)

            if step <= REQUIRED_STEP_PRECISION:
                print()
                print('weights')
                print(next_weights)
                new_policy = calculate_policy(next_weights)
                rewards.append(next_weights)
                policies.append(new_policy)
                policy_appended = True
                break

        if not policy_appended:

            probability = np.random.rand()

            if probability >= 0.5:

                new_policy = calculate_policy(next_weights)
                rewards.append(next_weights)

            else:

                new_policy = calculate_policy(current_weights)
                rewards.append(current_weights)

            policies.append(new_policy)

        final_reward = next_weights

    # print()
    # print('Final Reward Function')
    # print()
    # print(rewards)
    print()
    print('All rewards returned')
    for reward in rewards:
        print(reward)
        print()


main()
