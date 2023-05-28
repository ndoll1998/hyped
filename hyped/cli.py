import logging
import datasets
from argparse import ArgumentParser
from hyped.stages.prepare import main as prepare
from hyped.stages.train import main as train
from hyped.stages.test import main as test

def main():
    # set log level
    logging.basicConfig(level=logging.INFO)

    # create global parser
    parser = ArgumentParser(prog="hyped")
    # add distributed setup arguments
    parser.add_argument("-r", "--local_rank", type=int, default=-1, help="Local rank of process during distributed inference")

    # create stages sub-parsers
    stage_parsers = parser.add_subparsers(
        title="stages", help="Pipeline stages"
    )

    # prepare stage argument parser
    prepare_parser = stage_parsers.add_parser("prepare", description="Prepare dataset for training")
    prepare_parser.add_argument("-c", "--config", type=str, required=True, help="Path to run configuration file in .json format")
    prepare_parser.add_argument("-n", "--max-size", type=int, default=None, help="Maximum number of data points per split")
    prepare_parser.add_argument("-s", "--splits", type=str, nargs='*', default=[], help="Subset of data splits to prepare")
    prepare_parser.add_argument("-o", "--out-dir", type=str, required=True, help="Path to store prepared dataset in")
    prepare_parser.set_defaults(func=prepare)

    # train stage argument parser
    train_parser = stage_parsers.add_parser("train", description="Train Transformer model on prepared datasets")
    train_parser.add_argument("-c", "--config", type=str, required=True, help="Path to run configuration file in .json format")
    train_parser.add_argument("-d", "--data", type=str, nargs='+', required=True, help="Paths to prepared data dumps")
    train_parser.add_argument("-o", "--out-dir", type=str, default=None, help="Output directory, by default uses directoy specified in config")
    train_parser.set_defaults(func=train)

    # test stage argument parser
    test_parser = stage_parsers.add_parser("test", description="Evaluate trained model on prepared datasets")
    test_parser.add_argument("-c", "--config", type=str, required=True, help="Path to run configuration file in .json format")
    test_parser.add_argument("-m", "--model-ckpt", type=str, required=True, help="Path to fine-tuned model checkpoint")
    test_parser.add_argument("-d", "--data", type=str, nargs='+', required=True, help="Paths to prepared data dumps")
    test_parser.add_argument("-s", "--splits", type=str, nargs='+', default=[datasets.Split.TEST], help="Subset of data splits to prepare, defaults to test split")
    test_parser.add_argument("-o", "--out-dir", type=str, default=None, help="Output directory, by default saves metrics in checkpoint")
    test_parser.set_defaults(func=test)

    # parse arguments and run function
    args = vars(parser.parse_args())
    args.pop("func")(**args)
