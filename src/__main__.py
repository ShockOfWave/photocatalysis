import os
import argparse
from src import __version__
from src.data.make_dataset import read_and_process_data
from src.data.split_data import split_and_save_data
from src.train.tune_models import tune_all_estimators
from src.train.train_models import train_models
from src.utils import PATH_TO_DATA


def load_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        prog="Train and tune ML model",
        description="This script train and tune machine learning models for photocatalysis project",
    )

    parser.add_argument(
        '-t', '--tune',
        action='store_true',
        help='Activate tuning process'
    )

    parser.add_argument(
        '-f', '--fit',
        action='store_true',
        help='Activate training process'
    )

    parser.add_argument(
        '-d', '--data',
        action='store_true',
        help='Activate dataset creation process'
    )

    parser.add_argument(
        '-s', '--split',
        action='store_true',
        help='Activate data splitting process'
    )

    parser.add_argument(
        '-v', '--version',
        action='version',
        version="%(prog)s {}".format(__version__),
        help='Show the version number and exit',
    )

    args = parser.parse_args()

    return args


def main():
    args = load_args()

    if args.data:
        read_and_process_data(os.path.join(PATH_TO_DATA, 'raw', 'data.csv'))

    if args.split:
        split_and_save_data(os.path.join(PATH_TO_DATA, 'processed', 'data.csv'))

    if args.tune:
        tune_all_estimators()

    if args.fit:
        train_models()


if __name__ == '__main__':
    main()
