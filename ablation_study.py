import os
import sys
import argparse
import glob
import re

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from train import create_fully_connected_nn, calculate_mean_and_std, \
    prepare_dataset


def natural_sort(array):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return sorted(array, key=lambda key: [convert(c)
                                          for c in re.split('([0-9]+)', key)])


def prepare_kfold_cross_validation(dataset_path, num_splits=10):
    """
    Read the data from a csv file and split the data into a train and val split.

    Args:
        dataset_path (str): Path to the dataset .csv file.
        num_splits (int): Number of splits for k fold cross validation.

    Returns:
        kf (KFold): K-Folds cross-validator
        x (np.ndarray): X data.
        y (np.ndarray): y data.
    """
    assert dataset_path.endswith('.csv')
    df = pd.read_csv(dataset_path, header=None)

    x = df.iloc[:, :4].to_numpy()
    y = df.iloc[:, 4:].to_numpy()

    kf = KFold(n_splits=num_splits)

    return kf.split(x, y), x, y


def main(args):
    """
    Run ablation study to find the best model depending on the number of samples
    in the dataset. Result can be seen in the results.png image.
    Args:
        args: Command line arguments.
    """
    args = parse_arguments(args)

    # Prepare K-Fold cross validation of the original dataset
    kf, x, y = prepare_kfold_cross_validation('./dataset.csv',
                                              num_splits=args.num_splits)

    output_dir = './ablation_study_checkpoints'
    original_output_dir = os.path.join(output_dir, 'orig')
    generated_output_dir = os.path.join(output_dir, 'gen')

    results_orig = []
    results_gen = []
    dataset_sizes = []
    for fold_id, (train_index, test_index) in enumerate(kf):
        x_train, x_val = x[train_index], x[test_index]
        y_train, y_val = y[train_index], y[test_index]
        x_moments = calculate_mean_and_std(x_train)
        y_moments = calculate_mean_and_std(y_train)

        # Create and train model on original dataset
        model_orig = create_fully_connected_nn((4,),
                                               x_moments,
                                               y_moments,
                                               args.model_structure)
        model_orig.compile(optimizer=args.optimizer, loss=args.loss_func)

        model_orig.fit(
            x_train, y_train, batch_size=args.batch_size,
            epochs=args.num_epochs, validation_data=(x_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=args.early_stopping_patience),
                tf.keras.callbacks.ModelCheckpoint(original_output_dir,
                                                   save_best_only=True)
            ])

        # Load best checkpoint
        model_orig = tf.keras.models.load_model(original_output_dir)
        results_orig.append(model_orig.evaluate(x_val, y_val,
                                                batch_size=args.batch_size))

        results_gen_fold = []
        for generated_dataset_path in natural_sort(
                glob.glob('generated_datasets/*.csv')):

            dataset_size = int(generated_dataset_path.split(
                '-')[1].split('.csv')[0])

            if fold_id == 0:
                dataset_sizes.append(dataset_size)
            # Prepare generated dataset data
            x_train_gen, x_val_gen, y_train_gen, y_val_gen = \
                prepare_dataset(generated_dataset_path, val_size=args.val_size)

            x_moments_gen = calculate_mean_and_std(x_train_gen)
            y_moments_gen = calculate_mean_and_std(y_train_gen)

            # Create model for training on the generated data
            model_gen = create_fully_connected_nn((4,),
                                                  x_moments_gen,
                                                  y_moments_gen,
                                                  args.model_structure)
            model_gen.compile(optimizer=args.optimizer, loss=args.loss_func)

            model_gen.fit(
                x_train_gen, y_train_gen, batch_size=args.batch_size,
                epochs=args.num_epochs, validation_data=(x_val_gen, y_val_gen),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=args.early_stopping_patience),
                    tf.keras.callbacks.ModelCheckpoint(generated_output_dir,
                                                       save_best_only=True)
                ])

            # Load the best model
            model_gen = tf.keras.models.load_model(generated_output_dir)

            results_gen_fold.append(model_gen.evaluate(
                x_val, y_val, batch_size=args.batch_size))
        results_gen.append(results_gen_fold)

    results_orig = np.mean(results_orig)
    results_gen = np.mean(results_gen, axis=0)

    # Visualize the data.

    plt.plot(dataset_sizes, [results_orig]*len(dataset_sizes), color='b',
             label='Original dataset')
    plt.plot(dataset_sizes, results_gen, color='g', label='Generated dataset')

    plt.title('MAE loss on val set depending\non the number of samples '
              'in the dataset')

    plt.xlabel('Number of samples in the generated dataset')
    plt.ylabel('MAE averaged over different validation folds')
    plt.xscale('log')

    plt.legend()
    plt.savefig(args.output_name)


def parse_arguments(args):
    """
    Parse command line arguments.
    Args:
        args: Command line arguments.

    Returns:
        args: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size.')

    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs.')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use for training.')

    parser.add_argument('--loss_func', type=str, default='mae',
                        help='Loss function.')

    parser.add_argument('--output_name', type=str, default='./results.png',
                        help='Name of the output file.')

    parser.add_argument('--val_size', type=int, default=0.3,
                        help='Validation split size.')

    parser.add_argument('--early_stopping_patience', type=int, default=4,
                        help='Early stopping patience.')

    parser.add_argument('--num_splits', type=int, default=10,
                        help='Number of splits.')

    parser.add_argument('--model_structure', type=str, default=(20, 20, 2),
                        nargs='+', help='Model structure, i. e. number of '
                                        'neurons per layer.')

    return parser.parse_args(args)


if __name__ == '__main__':
    main(sys.argv[1:])
