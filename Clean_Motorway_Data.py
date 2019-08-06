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

    # todo start here. Clean all this up

    # data.dropna(inplace=True)
    def calculate_bins():

        return list(range(0, BIN_STEP_SIZE*NUMBER_OF_BINS, BIN_STEP_SIZE))

    bins = calculate_bins()
    trajectory = list()

    # calculate actions by checking steps in acceleration
    # speed up = jump in acceleration by some step size threshold
    # maintain = acceleration in some neighbourhood of 0
    # slow down = jump down in acceleration by some step size threshold

    # calculate initial state
    # record acceleration
    # continue until next threshold reached
    # record that action
    # record state the vehicle is now in
    # repeat

    acceleration_bins = [-1e99, -ACCELERATION_STEP_THRESHOLD, ACCELERATION_STEP_THRESHOLD, 1e99]

    states = (pd.cut(data['ttc'], bins, right=False, labels=False)
              .dropna()
              .reset_index(drop=True))  # set drop = False if you need original indices back

    actions = (pd.cut(data['xAcceleration'], acceleration_bins, right=False,
                      labels=['slow down', 'maintain speed', 'speed up'])
               .reset_index(drop=True))

    current_state = states[0]
    # if speed is negative, acceleration action needs to be flipped
    # otherwise, use action
    current_action = actions[0]

    trajectory.append(current_state)
    trajectory.append(current_action)

    print('states', states)
    print('actions', actions)
    print('trajectory', trajectory)


    return trajectory


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
MAINTAIN_SPEED_THRESHOLD = 10

# todo move methods around so that they make sense, correct dependencies

main()
