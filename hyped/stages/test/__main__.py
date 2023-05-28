import logging
import datasets
from .main import main
from argparse import ArgumentParser

# set log level
logging.basicConfig(level=logging.INFO)
# create argument parser
parser = ArgumentParser("prepare", description="Evaluate trained model on prepared datasets")
parser.add_argument("-c", "--config", type=str, required=True, help="Path to run configuration file in .json format")
parser.add_argument("-m", "--model-ckpt", type=str, required=True, help="Path to fine-tuned model checkpoint")
parser.add_argument("-d", "--data", type=str, nargs='+', required=True, help="Paths to prepared data dumps")
parser.add_argument("-s", "--splits", type=str, nargs='+', default=[datasets.Split.TEST], help="Subset of data splits to prepare, defaults to test split")
parser.add_argument("-o", "--out-dir", type=str, default=None, help="Output directory, by default saves metrics in checkpoint")
# parse arguments and run function
main(**vars(parser.parse_args()))
