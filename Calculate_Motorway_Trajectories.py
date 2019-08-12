import pandas as pd
import os

RAW_DATA_PATH = 'C:/Users/medkmin/highD-dataset-v1.0/data'           # path to raw data files
META_DATA_PATH = 'C:/Users/medkmin/highD-dataset-v1.0/data'     # path to meta data files
STATE_BIN_MAXIMUM = 60                                                        # maximum value for state bins
STATE_BIN_STEP_SIZE = 3                                                         # size of each state bin
ACCELERATION_STEP_THRESHOLD = 0.1   # threshold used to separate actions into slow down, maintain speed, and speed up
NO_STATE_CHANGE_THRESHOLD = 10      # number of steps in the same ttc to be considered another state in the trajectory


def main():
    """
    From motorway data for individual vehicles, we want to calculate
    the state-action trajectories of vehicles following other
    vehicles. We are not interested in lane changes. The states
    are determined by the Time To Collision (ttc) and the actions
    are slow down, maintain speed, and speed up which are
    determined by the acceleration of the vehicle. These state-action
    trajectories can then be used by Inverse Reinforcement Learning
    methods to create a reward function for these driving
    trajectories.

    We start by collecting all the file names for the raw
    and meta motorway data. For each pair, we find the
    vehicle id for all vehicles that do not change lanes.
    For each of these vehicles, we check if there is a
    possible collision in the data. This means there are ttcs
    that we can use that are appropriate i.e. they are
    positive and less than STATE BIN MAXIMUM. If there is,
    we pick the data that is relevant and then use it to
    calculate all the trajectories contained in that data.
    Depending on whether or not one or multiple trajectories
    are returned, we added them to the full list of trajectories.
    Once all of these are done for each vehicle for each pair
    of data files, we print and save the full list of trajectories.
    """
    def get_file_names_from_path(path, is_raw_data):
        """
        Collect all file names for files containing raw motorway data or meta track data.

        :param path:            path to folder where files are located
        :param is_raw_data:     boolean variable to determine if looking for raw or meta data
        :return:                list of file names for relevant data files
        """
        if is_raw_data:
            keyword = 'tracks.csv'
        else:
            keyword = 'tracksMeta.csv'

        files = list()
        for r, d, ff in os.walk(path):
            for file in ff:
                if keyword in file:
                    files.append(os.path.join(r, file))

        return files

    def get_data_from_file_name(file_name, is_raw_data):
        """
        Given a file name, read in data from that file. We import the relevant
        columns depending if the data is raw motorway data or meta track data.

        :param file_name:       path name to file
        :param is_raw_data:     boolean variable for raw or meta motorway data
        :return:                dataframe of imported data
        """
        if is_raw_data:
            columns_to_use = [1, 6, 8, 14, 16]
        else:
            columns_to_use = [0, 15]

        data = pd.read_csv(file_name, usecols=columns_to_use)

        return data

    def get_no_lane_change_vehicle_id(data):
        """
        Get the vehicle id numbers from the meta motorway data for all
        cars that do not change lanes.

        :param data:    meta motorway data
        :return:        series of id numbers for cars that do not change lanes
        """
        return data[data['numLaneChanges'] == 0]['id']

    def get_vehicle_data_from_id(data, current_id):
        """
        For a given vehicle id number, get the raw motorway data
        relevant to that vehicle.

        :param data:        raw motorway data
        :param current_id:  vehicle id being considered
        :return:            raw motorway data only relevant for that vehicle id
        """
        return data[data['id'] == current_id]

    def has_possible_collision(data):
        """
        Determine if the vehicle trajectory has a possible collision. In other words,
        does the vehicle have Times To Collision (ttc) that are positive, which ensures
        a collision could occur, and that is less than the STATE BIN MAXIMUM, a parameter
        to provide an upper limit on the ttcs that we consider. For example, if the ttc is
        over an hour, it may not be relevant to the vehicle dynamics.

        :param data:    raw motorway data
        :return:        boolean variable if any ttcs are both positive and less than STATE BIN MAXIMUM
        """
        return any((data['ttc'] > 0) & (data['ttc'] < STATE_BIN_MAXIMUM))

    def calculate_vehicle_trajectory(data):
        """
        Given the raw motorway data for a particular vehicle,
        we calculate all of the state-action trajectories for that
        data.

        We start by getting the ttc state bins and the acceleration action
        bins. If the data has one continuous trajectory and it has only
        one car being followed the whole time, we can calculate the single
        trajectory and return it with a flag that says we don't have multiple
        trajectories.

        If the data has one continuous trajectory but multiple cars being
        followed in front, then we get each trajectory for each car being
        followed and then return them all with a flag saying we do have
        multiple trajectories.

        If the data is not one continuous trajectory, we get the indexes
        for each continuous trajectory within the data. For each of these,
        we then check if the same car is being followed or not. If multiple
        cars, then we calculate the trajectory for each car being followed.
        Otherwise, we just calculate that continuous trajectory. We then
        return all trajectories with a flag saying we have multiple
        trajectories.

        :param data:    vehicle data being considered
        :return:        list of all trajectories within vehicle data
        """
        def get_ttc_bins():
            """
            Helper function to get bins for ttc states. The bins start at 0
            with step size STATE BIN STEP SIZE and end at STATE BIN MAXIMUM.

            :return:    list of ttc state bins
            """
            return list(range(0, STATE_BIN_MAXIMUM + STATE_BIN_STEP_SIZE, STATE_BIN_STEP_SIZE))

        def get_acceleration_bins():
            """
            Helper function to get acceleration action bins. The actions are
            slow down, maintain speed, and speed up. These actions are
            determined by the acceleration. If the acceleration is less than
            ACCELERATION STEP THRESHOLD, the vehicle is slowing down. If
            it is greater than ACCELERATION STEP THRESHOLD, the vehilce is
            speeding up. Otherwise, we say the vehicle is maintaining speed.

            :return:    bins for acceleration actions
            """
            return [-1e99, -ACCELERATION_STEP_THRESHOLD, ACCELERATION_STEP_THRESHOLD, 1e99]

        def one_continuous_trajectory(current_data):
            """
            Check if the vehicle data is one continuous trajectory. This means that
            no frames or time steps are skipped. They could be skipped if the car leaves
            the drone view, the car being followed changes lanes etc. If the
            journey is continuous, then the indexes will form a range from the minimum
            value to the maximum.

            :param current_data:    vehicle data being considered
            :return:                boolean variable if data is a continuous trajectory
            """
            return len(current_data.index) == len(list(range(min(current_data.index), max(current_data.index) + 1)))

        def following_same_car_whole_trajectory(current_data):
            """
            Determine if the vehicle is following the same car for
            the whole journey. If it is, the predcedingID data will
            only have one unique value.

            :param current_data:    current vehicle data being considered
            :return:                boolean variable if there is only one unique precedingID entry
            """
            return len(current_data.iloc[:, 4].unique()) == 1

        def get_single_trajectory(current_data):
            """
            For the current vehicle data, we calculate the single trajectory of
            the vehicle through states and actions. We begin by converting the
            ttcs into states and the xAcceleration into actions. Create a list
            for the trajectory and add the initial state and action.

            We then loop through the rest of the states and we keep a counter
            for how long the vehicle stays in the current state. If the next state is
            different from the current state, we add the new state and action at
            this new state to the trajectory. Else, if the vehicle is in the
            same state for NO STATE CHANGE THRESHOLD time steps, then we add
            the same state and the action at that state to the threshold. Otherwise,
            we increment the counter for the vehicle being in that state.

            Essentially, there are two things that can happen to add a new state
            to the trajectory. Either the state changes in the ttcs or the
            NO STATE CHANGE THRESHOLD is reached. Once the loop has passed through
            all states, return the trajectory.

            States = bins of ttcs
            Action 0 = slow down, Action 1 = maintain speed, Action = 2 = speed up

            :param current_data:    Current vehicle data being considered
            :return:                Trajectory of states and actions for vehicle data
            """
            def calculate_action(state_index):
                """
                There is a mistake in the raw data. When the acceleration of
                a vehicle is calculated when the vehicle has negative velocity,
                the sign of the acceleration is reversed. To see this, check the
                data for vehicle id 5 in 01_tracks.csv.

                This function corrects the action value when the velocity is
                negative.

                :param state_index:     index in state data being considered
                :return:                corrected action to be selected
                """
                if current_data.iloc[state_index, 1] < 0:

                    action = 2 - actions.iloc[state_index, 1]

                else:

                    action = actions.iloc[state_index, 1]

                return action

            def append_new_state_and_action(current_trajectory, state, action):
                """
                Helper function that adds the state and action to the trajectory.

                :param current_trajectory:  Current trajectory being considered
                :param state:               The state to be added
                :param action:              The action to be added
                :return:                    Current trajectory with state and action added
                """
                current_trajectory.append(state)
                current_trajectory.append(action)

                return current_trajectory

            states = (pd.cut(current_data['ttc'], ttc_bins, right=False, labels=False)
                      .dropna()
                      .reset_index(drop=False))

            actions = (pd.cut(current_data['xAcceleration'], acceleration_bins, right=False, labels=False)
                       .reset_index(drop=False))

            current_single_trajectory = list()

            current_state = states.iloc[0, 1]
            current_action = calculate_action(0)

            current_single_trajectory = append_new_state_and_action(current_single_trajectory,
                                                                    current_state, current_action)

            current_state_counter = 0

            for next_state_index in range(len(states.iloc[:, 1])):

                if states.iloc[next_state_index, 1] != current_state:

                    next_state = states.iloc[next_state_index, 1]
                    next_action = calculate_action(next_state_index)

                    current_single_trajectory = append_new_state_and_action(current_single_trajectory,
                                                                            next_state, next_action)

                    current_state = next_state
                    current_state_counter = 0

                elif current_state_counter == NO_STATE_CHANGE_THRESHOLD:

                    next_action = calculate_action(next_state_index)

                    current_single_trajectory = append_new_state_and_action(current_single_trajectory,
                                                                            current_state, next_action)

                    current_state_counter = 0

                else:

                    current_state_counter += 1

            return current_single_trajectory

        def get_multiple_front_car_trajectories(current_data):
            """
            If the vehicle has multiple cars in front of it, we want to calculate
            the trajectory for each of these cars. We loop through all front car
            ids and then calculate the single trajectory for that particular car
            in front. We add each of these trajectories to a list of all trajectories
            and return that.

            :param current_data:    current vehicle data being considered
            :return:                all trajectories for each car in front
            """
            all_front_car_ids = current_data.iloc[:, 4].unique()
            all_front_car_trajectories = list()

            for front_car_id in all_front_car_ids:

                data_with_current_front_car = current_data[current_data.iloc[:, 4] == front_car_id]
                trajectory_with_current_front_car = get_single_trajectory(data_with_current_front_car)
                all_front_car_trajectories.append(trajectory_with_current_front_car)

            return all_front_car_trajectories

        def get_single_trajectory_indexes():
            """
            When the vehicle data has multiple trajectories within it,
            we can partition the indexes into the individual single
            trajectory indexes.

            We start at the initial index and then loop through the
            rest of the indexes. If the next index is equal to the previous
            index plus one and not the final index, then we add the new
            index to the current trajectory. If the next index is not
            the previous index plus one, we are at the beginning of a new
            trajectory so we save the indexes and begin a new list.
            Otherwise, we are at the end of the list so we added the last
            index, add the list to the list of all trajectory indexes,
            and then we return the list of all trajectory indexes.

            :return:    list of indexes for each single trajectory within the indexes
            """
            all_trajectory_indexes = list()
            start_index = data.index[0]

            current_index_list = [start_index]
            previous_index = start_index

            for current_index in data.index[1:]:

                if (current_index == previous_index + 1) and (current_index != data.index[-1]):

                    current_index_list.append(current_index)
                    previous_index = current_index

                elif current_index < data.index[-1]:

                    all_trajectory_indexes.append(pd.Int64Index(current_index_list))
                    current_index_list = [current_index]
                    previous_index = current_index

                else:

                    current_index_list.append(current_index)
                    all_trajectory_indexes.append(pd.Int64Index(current_index_list))

            return all_trajectory_indexes

        ttc_bins = get_ttc_bins()
        acceleration_bins = get_acceleration_bins()

        if one_continuous_trajectory(data):

            if following_same_car_whole_trajectory(data):

                single_trajectory = get_single_trajectory(data)
                return single_trajectory, False

            else:

                trajectories = get_multiple_front_car_trajectories(data)
                return trajectories, True

        else:

            single_trajectory_indexes = get_single_trajectory_indexes()
            trajectories = list()

            for single_trajectory_index in single_trajectory_indexes:

                single_trajectory_data = data.loc[single_trajectory_index, :]

                if following_same_car_whole_trajectory(single_trajectory_data):

                    single_trajectory = get_single_trajectory(single_trajectory_data)
                    trajectories.append(single_trajectory)

                else:

                    current_trajectories = get_multiple_front_car_trajectories(single_trajectory_data)
                    trajectories.extend(current_trajectories)

            return trajectories, True

    all_raw_data_file_names = get_file_names_from_path(RAW_DATA_PATH, is_raw_data=True)
    all_meta_data_file_names = get_file_names_from_path(META_DATA_PATH, is_raw_data=False)

    zipped_raw_and_meta_file_names = list(zip(all_raw_data_file_names, all_meta_data_file_names))

    all_trajectories = list()

    for i, (raw_data_file_name, meta_data_file_name) in enumerate(zipped_raw_and_meta_file_names):

        all_raw_data = get_data_from_file_name(raw_data_file_name, is_raw_data=True)
        all_meta_data = get_data_from_file_name(meta_data_file_name, is_raw_data=False)

        no_lane_change_vehicle_id = get_no_lane_change_vehicle_id(all_meta_data)

        for j, vehicle_id in enumerate(no_lane_change_vehicle_id):

            print('file progress', (i+1)/len(zipped_raw_and_meta_file_names)*100, '%',
                  'vehicle progress', (j+1)/len(no_lane_change_vehicle_id)*100, '%',
                  'number of trajectories', len(all_trajectories))

            vehicle_data = get_vehicle_data_from_id(all_raw_data, vehicle_id)

            if has_possible_collision(vehicle_data):

                following_data = vehicle_data[(vehicle_data['ttc'] > 0) & (vehicle_data['ttc'] < STATE_BIN_MAXIMUM)]
                vehicle_trajectory, multiple_trajectories = calculate_vehicle_trajectory(following_data)

                if multiple_trajectories:
                    all_trajectories.extend(vehicle_trajectory)
                else:
                    all_trajectories.append(vehicle_trajectory)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print()
        print('all trajectories')
        print(all_trajectories)

    with open('trajectories.txt', 'w') as f:
        for trajectory in all_trajectories:
            f.write("%s\n" % trajectory)

    with open('details.txt', 'w') as f:
        f.write("%s\n" % STATE_BIN_MAXIMUM)
        f.write("%s\n" % STATE_BIN_STEP_SIZE)


main()
