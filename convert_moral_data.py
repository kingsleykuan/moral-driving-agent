#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

COLUMNS = [
    'ResponseID',
    'Intervention',
    'PedPed',
    'Barrier',
    'CrossingSignal',
    'Man',
    'Woman',
    'Pregnant',
    'Stroller',
    'OldMan',
    'OldWoman',
    'Boy',
    'Girl',
    'Homeless',
    'LargeWoman',
    'LargeMan',
    'Criminal',
    'MaleExecutive',
    'FemaleExecutive',
    'FemaleAthlete',
    'MaleAthlete',
    'FemaleDoctor',
    'MaleDoctor',
    'Dog',
    'Cat',
    'Saved',
]
NUMERIC_COLUMNS = COLUMNS[1:]


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert Moral Machine data.")

    parser.add_argument(
        '--input_data_path', default='data/moral_data.csv', type=str,
        help='Path to extracted Moral Machine data.')

    parser.add_argument(
        '--data_path_train', default='data/moral_data_train.npz', type=str,
        help='Path to output Moral Machine train NPZ.')

    parser.add_argument(
        '--data_path_val', default='data/moral_data_val.npz', type=str,
        help='Path to output Moral Machine val NPZ.')

    parser.add_argument(
        '--data_path_test', default='data/moral_data_test.npz', type=str,
        help='Path to output Moral Machine test NPZ.')

    parser.add_argument(
        '--train_size', default=0.8, type=float,
        help='Ratio of data to use for training.')

    parser.add_argument(
        '--val_size', default=0.1, type=float,
        help='Ratio of data to use for validation.')

    parser.add_argument(
        '--random_seed', default=0, type=int,
        help='Random seed.')

    return parser


def split_data(data, train_size=0.8, val_size=0.1, random_seed=0):
    data_train, data_val_test = train_test_split(
        data,
        train_size=train_size,
        random_state=random_seed,
        shuffle=True)

    data_val, data_test = train_test_split(
        data_val_test,
        train_size=val_size / (1.0 - train_size),
        random_state=random_seed,
        shuffle=True)

    return data_train, data_val, data_test


def convert_moral_dataframe_to_numpy(data):
    data = data.to_numpy()
    data_not_saved, data_saved = np.split(data, 2, axis=-1)
    return data_not_saved, data_saved


def save_moral_numpy(data_path, data_not_saved, data_saved):
    np.savez_compressed(
        data_path,
        data_not_saved=data_not_saved,
        data_saved=data_saved)


def convert_moral_data(
        input_data_path,
        data_path_train,
        data_path_val,
        data_path_test,
        train_size=0.8,
        val_size=0.1,
        random_seed=0):
    dtype = {column: np.uint8 for column in NUMERIC_COLUMNS}
    data = pd.read_csv(
        input_data_path, sep=',', header=0, usecols=COLUMNS, dtype=dtype)
    print(f"Read {len(data)} rows")

    # Split data into saved and not saved
    data_not_saved = data[data['Saved'] == 0]
    data_saved = data[data['Saved'] == 1]

    # Merge saved and not saved rows on ResponseID
    data = pd.merge(
        data_not_saved,
        data_saved,
        how='inner',
        on='ResponseID',
        sort=True,
        suffixes=('NotSaved', 'Saved'),
        validate='one_to_one')
    data.drop(
        columns=['ResponseID', 'SavedNotSaved', 'SavedSaved'], inplace=True)
    del data_not_saved, data_saved
    print(f"Merged data into {len(data)} rows")

    data_train, data_val, data_test = split_data(
        data,
        train_size=train_size,
        val_size=val_size,
        random_seed=random_seed)
    del data
    print("Split data into:")
    print(f"Train: {len(data_train)}")
    print(f"Val: {len(data_val)}")
    print(f"Test: {len(data_test)}")

    data_train = convert_moral_dataframe_to_numpy(data_train)
    data_val = convert_moral_dataframe_to_numpy(data_val)
    data_test = convert_moral_dataframe_to_numpy(data_test)

    save_moral_numpy(data_path_train, *data_train)
    save_moral_numpy(data_path_val, *data_val)
    save_moral_numpy(data_path_test, *data_test)


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    convert_moral_data(**vars(args))
