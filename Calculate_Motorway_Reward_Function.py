import numpy as np
import multiprocessing as mp
np.set_printoptions(linewidth=np.nan)

LEARNING_RATE = 0.1
MAXIMUM_WEIGHT_UPDATES = 100
MAXIMUM_NUMBER_OF_POLICIES = 1
REQUIRED_STEP_PRECISION = 0.01

REWARD_DISCOUNT_FACTOR = 1

P_FUNCTION_REWARD = 1
P_FUNCTION_PENALTY = 2

MAXIMUM_NUMBER_OF_POLICY_EPISODES = 10000
MAXIMUM_POLICY_ITERATIONS = 1000
LAMBDA = 0.9
EPSILON_RATIO_VALUE = 1000


# todo think about logic of code and try to make it as efficient as possible
# todo run on ARC
# todo rewrite with global variables instead of passing all variables
# todo debug terminal state code to make sure it works correctly
# todo add in randomness to create simulated episodes code. Ttc doesn't always change according to these actions.
# add randomness for changes in distance state and front vehicle speed
# todo possibly add randomness here to step model method
# todo explain in more detail what calculate policy actually creates
# todo figure out how to pick new action from action value function independent of state size
# todo rearrange methods so that they are in order, have correct scope for their usage


def get_feature_vector(feature_vector_number, state_sizes):
    """
    Helper function to get feature vector that is zero everywhere
    except a 1 in the given feature vector number index

    :param feature_vector_number:   index for the 1 in the vector
    :param state_sizes              size of the feature vector
    :return:                        feature vector with 1 in that index
    """
    feature_vector = np.zeros(state_sizes)
    feature_vector[feature_vector_number] = 1

    return feature_vector


def create_simulated_episodes(current_policy, state_dimensions, state_sizes, real_episode_lengths):
    """
    Create simulated episode in GridWorld from the given policy

    :param current_policy:          policy to use to create episode
    :param state_dimensions         number of dimensions of state vector
    :param state_sizes              number of possible values in each state variable
    :param real_episode_lengths     length of each real episode trajectory
    :return:                        episode created using given policy
    """
    def terminal_state(state, current_episode):
        """
        Check if the state has reached a terminal value, i.e. zero, max, or
        the length of the episode has reached the maximum length

        :param state:               current state of the episode
        :param current_episode:     the current episode being considered
        :return:                    boolean if a terminal state has been reached
        """
        if len(np.intersect1d(state, np.zeros(state_dimensions))) > 0:

            return True

        elif len(current_episode) == max_episode_length:

            return True

        elif len(np.intersect1d(state, state_sizes - np.ones(state_dimensions))) > 0:

            return True

        return False

    current_state = tuple(np.random.randint(state_size) for state_size in state_sizes)
    current_action = current_policy[current_state]
    episode = [current_state, current_action]
    max_episode_length = generate_max_episode_length(real_episode_lengths)

    while not terminal_state(current_state, episode):

        if current_action == 0:

            current_state += np.array([-1, 0, 0])

        elif current_action == 1:

            current_state += np.array([1, 0, 0])

        current_state = tuple(current_state)
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

    mean_episode_value = np.mean(episode_values)

    return mean_episode_value


def calculate_p_gradient(weights, current_policy, state_sizes, state_dimensions, real_episodes, real_episode_lengths):
    """
    Taking the partial derivative of the sum:

    sum_{i=1}^k p( V^{pi*}(s) - V^{pi_i}(s) )

    with respect to each weight alpha_n results
    in the following sum:

    sum_{i=1}^k p'( V^{pi*}(s) - V^{pi_i}(s) )
                    * ( V_n^{pi*}(s) - V_n^{pi_i}(s) )

    This method calculates p'( V^{pi*}(s) - V^{pi_i}(s) ).

    :param current_policy:          Policy used to compare against expert episodes
    :param weights                  Current weights to scale each state
    :param state_sizes              Number of possible values for each state variable
    :param state_dimensions         Size of state vector
    :param real_episodes            Trajectories from motorway data
    :param real_episode_lengths     Length of motorway data trajectories
    :return:                        Value of p' in gradient calculation
    """
    sum_product = np.zeros(state_sizes)
    feature_vector_index = tuple(np.zeros(state_dimensions).astype(int))

    while feature_vector_index[0] != state_sizes[0]:

        feature_vector = get_feature_vector(feature_vector_index, state_sizes)
        perfect_policy_weight_value = calculate_reward(real_episodes, feature_vector)

        simulated_episodes = create_simulated_episodes(current_policy, state_dimensions, state_sizes,
                                                       real_episode_lengths)
        policy_weight_value = calculate_reward(simulated_episodes, feature_vector)

        sum_product[feature_vector_index] = (weights[feature_vector_index]
                                             * (perfect_policy_weight_value - policy_weight_value))
        feature_vector_index = increment_index(feature_vector_index, state_sizes, state_dimensions)

    weight_sum = np.sum(sum_product)

    if weight_sum >= 0:
        return P_FUNCTION_REWARD
    else:
        return P_FUNCTION_PENALTY


def calculate_value_function_difference(feature_vector_number, current_policy, real_episodes,
                                        state_sizes, state_dimensions, real_episode_lengths):
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
    :param real_episodes            Trajectories from motorway data
    :param state_sizes              Number of possible values for each state variable
    :param state_dimensions         Length of state vector
    :param real_episode_lengths     Length of motorway data trajectories
    :return:                        Value of difference between policy values
    """
    feature_vector = get_feature_vector(feature_vector_number, state_sizes)
    perfect_policy_weight_value = calculate_reward(real_episodes, feature_vector)

    simulated_episodes = create_simulated_episodes(current_policy, state_dimensions, state_sizes, real_episode_lengths)
    policy_weight_value = calculate_reward(simulated_episodes, feature_vector)

    return perfect_policy_weight_value - policy_weight_value


def increment_index(current_index, state_sizes, state_dimensions):
    """
    Step the current index value to the next one in the state space

    :param current_index:       The index to be incremented
    :param state_sizes:         The number of possible values for each state variable
    :param state_dimensions:    The size of the state vector
    :return:                    The new incremented index
    """
    if type(current_index) == tuple:

        current_index = list(current_index)

    current_index[-1] += 1

    for pp in range(state_dimensions - 1, 0, -1):

        if current_index[pp] == state_sizes[pp]:
            current_index[pp] = 0
            current_index[pp - 1] += 1

    return tuple(current_index)


def calculate_gradient(state_sizes, current_policies, weights, state_dimensions, real_episodes,
                       real_episode_lengths):
    """
    Calculate the direction of the gradient to update the weights
    in order to find the weights that maximise the sum of the
    differences between the real episode values and random policy values.

    In particular, the sum we want to find weights for is:

    sum_{i=1}^k p( V^{pi*}(s) - V^{pi_i}(s) )

    where abs(alpha_i) <= 1 for all weights alpha_i
    and p(x) = x if x >= 0 and p(x) = 2x if x < 0, which
    is a penalty term for when V^{pi*}(s) < V^{pi_i}(s).

    We use multiprocessing here to speed up the run time.

    Full details can be found in section 5 of
    'Algorithms for Inverse Reinforcement Learning' by Ng and Russel
    https://ai.stanford.edu/~ang/papers/icml00-irl.pdf.

    :param weights:                 the current values for the weights
    :param current_policies:        the set of policies currently being optimised over
    :param state_sizes              The number of possible values for each state variable
    :param state_dimensions         The length of the state vector
    :param real_episodes            The trajectories from the motorway data
    :param real_episode_lengths     The length of the motorway trajectories
    :return:                        list of gradient components for each weight
    """
    state_indexes = list(np.ndindex(tuple(state_sizes)))
    other_params = [current_policies, weights, state_sizes, state_dimensions, real_episodes, real_episode_lengths]
    params = [[state_index] + other_params for state_index in state_indexes]
    gradient = pool.starmap(calculate_gradient_for_state_index, params)

    return gradient


def calculate_gradient_for_state_index(index, current_policies, weights, state_sizes, state_dimensions, real_episodes,
                                       real_episode_lengths):
    """
    Helper function for multiprocessing. This function is called in calculate_gradient
    using the starmap multiprocessing method.

    :param index:                   Current state index being considered
    :param current_policies:        List of policies already calculated
    :param weights:                 Weights used to scale states
    :param state_sizes:             Number of possible values for each state variable
    :param state_dimensions:        Length of state vector
    :param real_episodes:           Trajectories from motorway data
    :param real_episode_lengths:    Length of motorway trajectories
    :return:                        Gradient calculation for each parameter set
    """
    return sum([calculate_p_gradient(weights, policy, state_sizes, state_dimensions, real_episodes,
                                     real_episode_lengths)
                * calculate_value_function_difference(index, policy, real_episodes, state_sizes, state_dimensions,
                                                      real_episode_lengths)
                for policy in current_policies])


def step_model(state, action, weights, state_sizes):
    """
    Step through the model given a state and an action.
    Return the new state, immediate reward, and whether or
    not it is a terminal state.

    If the agent moves off Gridworld or back to the previous
    state, there is a negative reward.

    Sub methods filter through different values for each state
    element and then consider the appropriate action and reward.

    Action = 0 (slow down)
    Action = 1 (speed up)

    :param state:           Current state of the agent
    :param action:          Current action the agent takes
    :param weights:         Weights that approximate rewards for various actions
    :param state_sizes      Number of possible values for each state variable
    :return:                New state, immediate reward, and if new state is a terminal state
    """
    def step_state_middle(current_state):

        if action == 0:

            current_state = tuple(map(sum, zip(current_state, (-1, 0, 0))))

        elif action == 1:

            current_state = tuple(map(sum, zip(current_state, (1, 0, 0))))

        return tuple(current_state), weights[current_state]

    def step_state_zero(current_state):

        if action == 0:

            pass

        elif action == 1:

            current_state = tuple(map(sum, zip(current_state, (1, 0, 0))))

        return tuple(current_state), weights[current_state]

    def step_state_max(current_state):

        if action == 0:

            current_state = tuple(map(sum, zip(current_state, (-1, 0, 0))))

        elif action == 1:

            pass

        return tuple(current_state), weights[current_state]

    if (state[0] > 0) and (state[0] < state_sizes[0] - 1):

        return step_state_middle(state)

    elif state[0] == 0:

        return step_state_zero(state)

    elif state[0] == state_sizes[0] - 1:

        return step_state_max(state)


def calculate_policy(weights, state_sizes, number_of_actions):
    """
    The weights that we calculate can also be
    interpreted as rewards. For the given set of weights,
    we calculate a policy that receives a maximal total reward
    through an episode of following the policy.

    We calculate this policy using the SARSA(lambda) method
    found in these lecture notes:

    http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf

    :param weights:             approximate values for rewards used to calculate policy
    :param state_sizes:         Number of possible values for each state variable
    :param number_of_actions:   Number of different actions agents can take
    :return:                    policy that maps states to actions in a way that maximises
                                the total reward received
    """
    policy = np.random.randint(0, number_of_actions, state_sizes)

    action_value_function = np.zeros(tuple(np.append(state_sizes, number_of_actions).astype(int)))
    state_action_counter = np.zeros(tuple(np.append(state_sizes, number_of_actions).astype(int)))
    state_counter = np.zeros(tuple(state_sizes.astype(int)))

    for m in range(MAXIMUM_NUMBER_OF_POLICY_EPISODES):

        print('policy episode', m)

        eligibility_traces = np.zeros(tuple(np.append(state_sizes, number_of_actions).astype(int)))

        state = tuple(np.random.randint(state_size) for state_size in state_sizes)
        action = policy[state]

        new_state, reward = step_model(state, action, weights, state_sizes)
        counter = 0

        while counter < MAXIMUM_POLICY_ITERATIONS:

            counter += 1
            state_counter[state] += 1

            state_action_pair = state + tuple([action])
            state_action_counter[state_action_pair] += 1

            epsilon = EPSILON_RATIO_VALUE / (state_counter[state] + EPSILON_RATIO_VALUE)
            explore_probability = np.random.rand()

            if explore_probability <= epsilon:

                policy[new_state] = np.random.randint(0, number_of_actions)

            else:

                policy[new_state] = np.argmax(action_value_function[new_state[0], new_state[1], new_state[2], :])

            new_action = policy[new_state]

            delta = (reward + action_value_function[new_state + tuple([new_action])]
                     - action_value_function[state_action_pair])

            eligibility_traces[state_action_pair] += 1

            action_value_function[state_action_pair] += (delta * eligibility_traces[state_action_pair]
                                                         / state_action_counter[state_action_pair])
            eligibility_traces[state_action_pair] *= LAMBDA * REWARD_DISCOUNT_FACTOR

            state = new_state
            action = new_action

            new_state, reward = step_model(state, action, weights, state_sizes)

    return policy


def generate_initial_policy(state_sizes):
    """
    A simple policy used to begin the inverse reinforcement learning
    method. When we are calculating the weights using gradient
    descent, we want to maximise the value function difference
    between the observed behaviour and any other policy. This random
    policy is the first policy we begin with.

    Action 0 = slow down

    :return:    policy filled with all 0s
    """
    policy = np.zeros(state_sizes)

    return policy


def create_real_episodes():
    """
    Load trajectories from motorway data.

    :return: episodes for IRL method to learn from
    """
    trajectories = list()
    trajectory_lengths = list()

    with open('trajectories.txt', 'r') as g:

        for line in g:

            trajectories.append(eval(line))
            trajectory_lengths.append(len(eval(line)) // 2)

    return trajectories, trajectory_lengths


def generate_max_episode_length(real_episode_lengths):
    """
    Generate a random episode length to be used in
    policy calculation.

    :param real_episode_lengths:    Lengths of motorway trajectories
    :return:                        Random integer for trajectory length
    """

    return np.random.randint(min(real_episode_lengths), max(real_episode_lengths) + 1)


def get_details():
    """
    Load motorway trajectory details from file.

    :return: Number of dimensions in state data, size of state variables,
             step size in data, maximum values, and number of actions
    """
    with open('details.txt', 'r') as ff:

        details = ff.readlines()

    actions = int(details[0])
    number_of_dimensions = int(details[1])
    sizes = np.zeros(number_of_dimensions).astype(int)
    step_size = np.zeros(number_of_dimensions).astype(int)
    maximum = np.zeros(number_of_dimensions).astype(int)

    for j in range(number_of_dimensions):

        step_size[j] = int(details[2 * j + 2])
        maximum[j] = int(details[2 * j + 3])
        sizes[j] = int(len(range(-maximum[j] - step_size[j], maximum[j] + step_size[j], step_size[j])))

    return number_of_dimensions, sizes, step_size, maximum, actions


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
    state_dimensions, state_sizes, state_step_sizes, state_maximums, number_of_actions = get_details()
    real_episodes, real_episode_lengths = create_real_episodes()

    random_policy = generate_initial_policy(state_sizes)
    policies = [random_policy]
    rewards = []

    while len(policies) - 1 < MAXIMUM_NUMBER_OF_POLICIES:

        next_weights = np.zeros(state_sizes)
        current_weights = np.zeros(state_sizes)
        policy_appended = False

        for i in range(MAXIMUM_WEIGHT_UPDATES):

            print('number of weight updates', i)

            current_weights = next_weights
            gradients = calculate_gradient(state_sizes, policies, current_weights, state_dimensions, real_episodes,
                                           real_episode_lengths)

            weights_change = np.reshape([LEARNING_RATE * gradient for gradient in gradients], state_sizes)
            next_weights = current_weights + weights_change

            next_weights = np.minimum(next_weights, np.ones(len(next_weights)))
            next_weights = np.maximum(next_weights, -1*np.ones(len(next_weights)))

            step = np.linalg.norm(next_weights - current_weights)

            if step <= REQUIRED_STEP_PRECISION:

                new_policy = calculate_policy(next_weights, state_sizes, number_of_actions)
                rewards.append(next_weights)
                policies.append(new_policy)
                policy_appended = True
                break

        if not policy_appended:

            probability = np.random.rand()

            if probability >= 0.5:

                new_policy = calculate_policy(next_weights, state_sizes, number_of_actions)
                rewards.append(next_weights)

            else:

                new_policy = calculate_policy(current_weights, state_sizes, number_of_actions)
                rewards.append(current_weights)

            policies.append(new_policy)

    with open('reward_function.txt', 'w') as f:

        for this_reward in rewards:

            f.write("%s\n" % this_reward)


if __name__ == '__main__':

    pool = mp.Pool(processes=mp.cpu_count())
    main()
