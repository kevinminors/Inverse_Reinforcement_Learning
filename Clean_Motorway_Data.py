import numpy as np
import pandas as pd


def import_motorway_data():

    data = pd.read_csv('01_tracks.csv',
                       usecols=[6, 10, 15],
                       dtype=np.float)
    return data


def add_time_to_collision_data(data):

    # todo analyse using columns instead of rows
    # might be faster
    # todo consider removing case when forward vehicle is faster
    # use ttc column in raw data

    def calculate_time_to_collision(row):

        if row.xVelocity - row.precedingXVelocity == 0:

            return -999

        elif row.precedingXVelocity == 0:

            return np.nan

        else:

            return row.frontSightDistance / (row.xVelocity - row.precedingXVelocity)

    data.dropna(inplace=True)
    data['xVelocity'] = abs(data['xVelocity'])
    data['precedingXVelocity'] = abs(data['precedingXVelocity'])
    data['time_to_collision'] = data.apply(calculate_time_to_collision, axis=1)

    return data


def calculate_bins(data):

    bin_minimum = int(np.floor(min(data['time_to_collision'])))
    bin_maximum = int(np.ceil(max(data['time_to_collision'])))
    bin_size = int(bin_maximum // NUMBER_OF_BINS)

    return list(range(bin_minimum, bin_maximum, bin_size))


def add_state_data(data):

    bins = calculate_bins(data)
    data['state'] = pd.cut(data['time_to_collision'], bins, right=False)
    return data


def main():

    raw_motorway_data = import_motorway_data()
    time_to_collision_data = add_time_to_collision_data(raw_motorway_data)
    state_data = add_state_data(time_to_collision_data)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(state_data[state_data['time_to_collision'] < 20])

    # calculate trajectories for each vehicle id in state data
    # using state = time to collision and
    # action = slow down, maintain speed, speed up


NUMBER_OF_BINS = 100

# todo remove trajectories that contain a lane change
# todo the ttc column in data stands for time to collision


main()
