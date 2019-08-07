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

    def one_continuous_trajectory(indexes):

        return len(indexes) == len(list(range(min(indexes), max(indexes) + 1)))

    def append_new_state_and_action(current_trajectory, state, action):

        current_trajectory.append(state)
        current_trajectory.append(action)

        return current_trajectory

    def calculate_action(state_index):

        if data.iloc[state_index, 6] < 0:

            action = 2 - actions.iloc[state_index, 1]

        else:

            action = actions.iloc[state_index, 1]

        return action

    ttc_bins = get_ttc_bins()
    acceleration_bins = get_acceleration_bins()

    states = (pd.cut(data['ttc'], ttc_bins, right=False, labels=False)
              # remove labels=False if you want to see intervals
              .dropna()
              .reset_index(drop=False))  # set drop = False if you need original indices back

    actions = (pd.cut(data['xAcceleration'], acceleration_bins, right=False, labels=False)
               .reset_index(drop=False))

    if one_continuous_trajectory(states.iloc[:, 0]):

        trajectory = list()

        current_state = states.iloc[0, 1]
        current_action = calculate_action(0)

        trajectory = append_new_state_and_action(trajectory, current_state, current_action)

        current_state_counter = 0

        for next_state_index in range(len(states.iloc[:, 1])):

            if states.iloc[next_state_index, 1] != current_state:

                next_state = states.iloc[next_state_index, 1]
                next_action = calculate_action(next_state_index)

                trajectory = append_new_state_and_action(trajectory, next_state, next_action)

                current_state = next_state
                current_state_counter = 0

            elif current_state_counter == NO_STATE_CHANGE_THRESHOLD:

                next_action = calculate_action(next_state_index)

                trajectory = append_new_state_and_action(trajectory, current_state, next_action)

                current_state_counter = 0

            elif (states.iloc[next_state_index, 1] == current_state
                    and current_state_counter < NO_STATE_CHANGE_THRESHOLD):

                current_state_counter += 1

            else:

                print('error 1')

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print()
            print('data')
            print(data[['xVelocity', 'xAcceleration', 'ttc']])
            print()
            print('states')
            print(states)
            print()
            print('actions')
            print(actions)
            print()
            print('trajectory')
            print(trajectory)

        # todo step through these trajectories to make sure nothing weird is happening

        return trajectory

    else:

        # break states down into seperate trajectories
        # calculate trajectory for each piece
        print(states)
        trajectories = list()

        # write above code as a method and then call it here for each bit of trajectory


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
                vehicle_trajectory = calculate_vehicle_trajectory(following_data)

                # check if return value is one list or if multiple lists of lists


                # print(vehicle_trajectory)
                # calculate actions
                # calculate trajectories
                # add trajectories

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(all_trajectories)


RAW_DATA_PATH = 'M:/GitHub/Inverse_Reinforcement_Learning/Track_Data'
META_DATA_PATH = 'M:/GitHub/Inverse_Reinforcement_Learning/Track_Meta_Data'
NUMBER_OF_BINS = 100
BIN_STEP_SIZE = 3
ACCELERATION_STEP_THRESHOLD = 0.1
NO_STATE_CHANGE_THRESHOLD = 10

# todo move methods around so that they make sense, correct dependencies
# todo do action check before during bin calculation or at some other time. THINK ABOUT THIS
# todo reduce data we are reading in and then use iloc everywhere
# todo add check if car in front moved and ttc is calculated for a car further in front. need new traj for this
# when preceding id changes during trajectory

main()
