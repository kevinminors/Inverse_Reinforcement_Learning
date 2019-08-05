import numpy as np
import pandas as pd


def import_motorway_data():

    data = pd.read_csv('01_tracks.csv',
                       usecols=[6, 10, 15],
                       dtype=np.float)
    return data


def add_time_to_collision_data(data):

    def calculate_time_to_collision(row):

        if abs(row.xVelocity) - abs(row.precedingXVelocity) == 0:

            return np.nan

        else:

            return row.frontSightDistance / (abs(row.xVelocity) - abs(row.precedingXVelocity))

    data['time_to_collision'] = data.apply(calculate_time_to_collision, axis=1)

    return data


def calculate_bins(data):

    bin_minimum = int(np.floor(min(data['time_to_collision'])))
    bin_maximum = int(np.ceil(max(data['time_to_collision'])))
    bin_size = int(bin_maximum // NUMBER_OF_BINS)

    return [bin_minimum] + list(range(0, bin_maximum, bin_size))


def add_state_data(data, bins):

    data['state'] = pd.cut(data['time_to_collision'], bins, right=False)
    return data


def main():

    raw_motorway_data = import_motorway_data()
    time_to_collision_data = add_time_to_collision_data(raw_motorway_data)
    bins = calculate_bins(time_to_collision_data)
    state_data = add_state_data(time_to_collision_data, bins)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(state_data.head(50))


NUMBER_OF_BINS = 100

# todo remove trajectories that contain a lane change
# todo remove trajectories where precedingXVelocity is 0,
# that just means the car is approaching the boundary

main()
