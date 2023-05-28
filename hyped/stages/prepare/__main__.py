import logging
from .main import main
from argparse import ArgumentParser

# set log level
logging.basicConfig(level=logging.INFO)
# create argument parser
parser = ArgumentParser("prepare", description="Prepare dataset for training")
parser.add_argument("-c", "--config", type=str, required=True, help="Path to run configuration file in .json format")
parser.add_argument("-n", "--max-size", type=int, default=None, help="Maximum number of data points per split")
parser.add_argument("-s", "--splits", type=str, nargs='*', default=[], help="Subset of data splits to prepare")
parser.add_argument("-o", "--out-dir", type=str, required=True, help="Path to store prepared dataset in")

# parse arguments and run function
main(**vars(parser.parse_args()))
