import argparse
import os
import sys

import numpy as np


def generate_data(s1, s2, s3, s4):
    """

    Args:
        s1 (np.ndarray): Sensor input data.
        s2 (np.ndarray): Sensor input data.
        s3 (np.ndarray): Sensor input data.
        s4 (np.ndarray): Sensor input data.

    Returns:
        np.ndarray: Generated dataset as a numpy array.
    """

    c = s1 + s2 ** 2 + 2 * s4
    f = s1 + s2 * 0.5 + 3 * s4 + 30

    s1, s2, s3, s4, c, f = [np.expand_dims(array, axis=1)
                            for array in [s1, s2, s3, s4, c, f]]

    return np.concatenate([s1, s2, s3, s4, c, f], axis=1)


def sample_input_domain(num_samples):
    """
    Generate random samples of the input domain.
    Args:
        num_samples (int): Number of samples to generate.

    Returns:
        tuple of np.ndarray: Sensors data.
    """
    s1 = np.random.random(num_samples) * 10
    s2 = np.random.random(num_samples) * 2 - 5
    s3 = np.random.random(num_samples)
    s4 = np.random.random(num_samples) * 30 + 20
    return s1, s2, s3, s4


def main(args):
    """
    Generate additional data.

    Args:
        args: Command line arguments.
    """

    args = parse_arguments(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Generating datasets with sizes:', args.dataset_sizes)
    for size in args.dataset_sizes:
        s1, s2, s3, s4 = sample_input_domain(size)
        data = generate_data(s1, s2, s3, s4)
        np.savetxt(os.path.join(args.output_dir, args.dataset_name + '-' +
                                str(size) + '.csv'),
                   data,
                   delimiter=",")
    print('Done')


def parse_arguments(args):
    """
    Parse command line arguments.
    Args:
        args: Command line arguments.

    Returns:
        args: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str,
                        default='generated_dataset',
                        help='Generated dataset name.')

    parser.add_argument('--output_dir', type=str,
                        default='./generated_datasets',
                        help='Model version.')

    parser.add_argument('--dataset_sizes', type=int, nargs='+',
                        default=(200, 500, 1000, 2000, 5000, 10000, 20000,
                                 50000, 100000, 200000),
                        help='Batch size.')

    return parser.parse_args(args)


if __name__ == '__main__':
    main(sys.argv[1:])
