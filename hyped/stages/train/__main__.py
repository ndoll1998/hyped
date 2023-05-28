import logging
from .main import main
from argparse import ArgumentParser

# set log level
logging.basicConfig(level=logging.INFO)
# create argument parser
parser = ArgumentParser("prepare", description="Train Transformer model on prepared datasets")
parser.add_argument("-c", "--config", type=str, required=True, help="Path to run configuration file in .json format")
parser.add_argument("-d", "--data", type=str, nargs='+', required=True, help="Paths to prepared data dumps")
parser.add_argument("-o", "--out-dir", type=str, default=None, help="Output directory, by default uses directoy specified in config")
parser.add_argument("-r", "--local_rank", type=int, default=-1, help="Local rank of process during distributed training")
# parse arguments and run function
main(**vars(parser.parse_args()))
