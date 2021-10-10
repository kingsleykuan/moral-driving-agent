#!/usr/bin/env python3

import argparse

import pandas as pd
from tqdm import tqdm

COLUMNS = [
    'ResponseID',
    'UserCountry3',
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
NUMERIC_COLUMNS = COLUMNS[2:]


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Extract Moral Machine data.")

    parser.add_argument(
        '--input_data_path', default='data/SharedResponses.csv', type=str,
        help='Path to Moral Machine SharedResponses.csv.')

    parser.add_argument(
        '--output_data_path', default='data/moral_data.csv', type=str,
        help='Path to output CSV.')

    parser.add_argument(
        '--countries', nargs='*', default=None, type=str,
        help='Countries of responses to extract.')

    parser.add_argument(
        '--chunksize', default=100000, type=int,
        help='Chunksize to read and write files.')

    return parser


def extract_moral_data(
        input_data_path,
        output_data_path,
        countries=None,
        chunksize=None):
    # Disable SettingWithCopyWarning
    pd.options.mode.chained_assignment = None

    # Read CSV file with pandas in chunks
    header = True
    mode = 'w'
    with pd.read_csv(
            input_data_path,
            sep=',',
            header=0,
            usecols=COLUMNS,
            chunksize=chunksize,
            low_memory=False) as reader:
        pbar = tqdm(reader, desc="Rows Processed: 0")
        for i, chunk in enumerate(pbar, 1):
            # Filter by countries
            if countries is not None:
                chunk = chunk[chunk['UserCountry3'].isin(countries)]

            # Convert numeric columns and drop nan values
            chunk[NUMERIC_COLUMNS] = chunk[NUMERIC_COLUMNS].apply(
                pd.to_numeric, errors='coerce')
            chunk = chunk.dropna()
            chunk[NUMERIC_COLUMNS] = chunk[NUMERIC_COLUMNS].apply(
                pd.to_numeric, downcast='integer')

            # Reorder columns and save CSV in chunks
            chunk = chunk.reindex(columns=COLUMNS)
            chunk.to_csv(
                output_data_path,
                sep=',',
                header=header,
                index=False,
                mode=mode)
            header = False
            mode = 'a'

            pbar.set_description(f"Rows Processed: {i * chunksize}")


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    extract_moral_data(**vars(args))
