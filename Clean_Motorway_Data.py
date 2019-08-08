import numpy as np
import pandas as pd
import os


def get_file_names_from_path(path):

    files = []
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    return files


def get_data_from_file_name(file_name):

    # todo reduce imports based on what we are importing and what data we eventually need
    # discard anything unnecessary
    data = pd.read_csv(file_name)  # , usecols=[6, 12, 14, 15], dtype=np.float)
    return data


def get_no_lane_change_vehicle_id(data):

    return data[data['numLaneChanges'] == 0]['id']


def get_vehicle_data_from_id(data, vehicle_id):

    return data[data['id'] == vehicle_id]


def has_possible_collision(data):

    return any(data['ttc'] > 0)


def calculate_vehicle_trajectory(data):

    def get_ttc_bins():

        return list(range(0, BIN_STEP_SIZE*NUMBER_OF_BINS, BIN_STEP_SIZE))

    def get_acceleration_bins():

        return [-1e99, -ACCELERATION_STEP_THRESHOLD, ACCELERATION_STEP_THRESHOLD, 1e99]

    def one_continuous_trajectory(current_data):

        return len(current_data.index) == len(list(range(min(current_data.index), max(current_data.index) + 1)))

    def append_new_state_and_action(current_trajectory, state, action):

        current_trajectory.append(state)
        current_trajectory.append(action)

        return current_trajectory

    def following_same_car_whole_trajectory(current_data):

        return len(current_data.iloc[:, 16].unique()) == 1

    def get_single_trajectory(current_data):

        def calculate_action(state_index):

            if current_data.iloc[state_index, 6] < 0:

                action = 2 - actions.iloc[state_index, 1]

            else:

                action = actions.iloc[state_index, 1]

            return action

        states = (pd.cut(current_data['ttc'], ttc_bins, right=False, labels=False)
                  # remove labels=False if you want to see intervals
                  .dropna()
                  .reset_index(drop=False))  # set drop = False if you need original indices back

        actions = (pd.cut(current_data['xAcceleration'], acceleration_bins, right=False, labels=False)
                   .reset_index(drop=False))

        single_trajectory = list()

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print()
            print('data')
            print(current_data[['xVelocity', 'xAcceleration', 'ttc']])
            print()
            print('states')
            print(states)
            print()
            print('actions')
            print(actions)

        current_state = states.iloc[0, 1]
        current_action = calculate_action(0)

        single_trajectory = append_new_state_and_action(single_trajectory, current_state, current_action)

        current_state_counter = 0

        for next_state_index in range(len(states.iloc[:, 1])):

            if states.iloc[next_state_index, 1] != current_state:

                next_state = states.iloc[next_state_index, 1]
                next_action = calculate_action(next_state_index)

                single_trajectory = append_new_state_and_action(single_trajectory, next_state, next_action)

                current_state = next_state
                current_state_counter = 0

            elif current_state_counter == NO_STATE_CHANGE_THRESHOLD:

                next_action = calculate_action(next_state_index)

                single_trajectory = append_new_state_and_action(single_trajectory, current_state, next_action)

                current_state_counter = 0

            elif (states.iloc[next_state_index, 1] == current_state
                  and current_state_counter < NO_STATE_CHANGE_THRESHOLD):

                current_state_counter += 1

            else:

                print('error 1')

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print()
            print('single_trajectory')
            print(np.reshape(single_trajectory, (len(single_trajectory)//2, 2)))

        # todo step through these trajectories to make sure nothing weird is happening

        return single_trajectory

    def get_multiple_front_car_trajectories(current_data):

        all_front_car_ids = current_data.iloc[:, 16].unique()
        current_trajectories = list()

        for front_car_id in all_front_car_ids:
            data_with_current_front_car = current_data[current_data.iloc[:, 16] == front_car_id]
            trajectory_with_current_front_car = get_single_trajectory(data_with_current_front_car)
            current_trajectories.append(trajectory_with_current_front_car)

        return current_trajectories

    def get_single_trajectory_indexes():

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

            trajectory = get_single_trajectory(data)
            return trajectory, False

        else:

            trajectories = get_multiple_front_car_trajectories(data)
            return trajectories, True

    else:

        single_trajectory_indexes = get_single_trajectory_indexes()
        trajectories = list()

        for single_trajectory_index in single_trajectory_indexes:

            single_trajectory_data = data.loc[single_trajectory_index, :]

            if following_same_car_whole_trajectory(single_trajectory_data):

                trajectory = get_single_trajectory(single_trajectory_data)
                trajectories.append(trajectory)

            else:

                current_trajectories = get_multiple_front_car_trajectories(single_trajectory_data)
                trajectories.extend(current_trajectories)

        return trajectories, True


def main():

    all_raw_data_file_names = get_file_names_from_path(RAW_DATA_PATH)
    all_meta_data_file_names = get_file_names_from_path(META_DATA_PATH)

    zipped_raw_and_meta_file_names = list(zip(all_raw_data_file_names, all_meta_data_file_names))

    all_trajectories = list()

    for raw_data_file_name, meta_data_file_name in zipped_raw_and_meta_file_names:

        all_raw_data = get_data_from_file_name(raw_data_file_name)
        all_meta_data = get_data_from_file_name(meta_data_file_name)

        no_lane_change_vehicle_id = get_no_lane_change_vehicle_id(all_meta_data)

        for vehicle_id in no_lane_change_vehicle_id:

            vehicle_data = get_vehicle_data_from_id(all_raw_data, vehicle_id)

            if has_possible_collision(vehicle_data):

                # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                #     print(vehicle_data)
                following_data = vehicle_data[vehicle_data['ttc'] > 0]
                vehicle_trajectory, multiple_trajectories = calculate_vehicle_trajectory(following_data)

                # check if return value is one list or if multiple lists of lists

                # todo check trajs are added correctly
                if multiple_trajectories:
                    all_trajectories.extend(vehicle_trajectory)
                else:
                    all_trajectories.append(vehicle_trajectory)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(all_trajectories)


RAW_DATA_PATH = 'M:/GitHub/Inverse_Reinforcement_Learning/Track_Data'
META_DATA_PATH = 'M:/GitHub/Inverse_Reinforcement_Learning/Track_Meta_Data'
NUMBER_OF_BINS = 10000000
BIN_STEP_SIZE = 3
ACCELERATION_STEP_THRESHOLD = 0.1
NO_STATE_CHANGE_THRESHOLD = 10

# todo move methods around so that they make sense, correct dependencies
# todo do action check before during bin calculation or at some other time. THINK ABOUT THIS
# todo reduce data we are reading in and then use iloc everywhere
# todo calculate state bins by calculating maximum ttc in all files
# or add condition where we throw away all data with ttcs bigger than our biggest bin
# if the time to collision is over an hour, we dont need it

main()
