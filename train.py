import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def create_fully_connected_nn(input_shape, x_moments, y_moments,
                              model_structure=(20, 20, 2)):
    """
    Create fully connected neural network with the hidden and output layers
    described with the model_structure parameter.

    Args:
        input_shape (tuple of ints): Input shape.
        model_structure (tuple of ints): Structure of the hidden and output
            layers. Number of neurons per each layer.
        x_moments (tuple of np.ndarray): Mean and std values of input data.
        y_moments (tuple of np.ndarray): Mean and std values of output data.

    Returns:
        model (tf.keras.Model): Fully connected neural network as a Keras Model.
    """
    x_mean, x_std = x_moments
    y_mean, _ = y_moments
    inputs = tf.keras.Input(input_shape, name='sensor_inputs')
    out = (inputs - x_mean) / x_std

    for num_neurons in model_structure:
        out = tf.keras.layers.Dense(num_neurons)(out)

    out += y_mean

    model = tf.keras.Model(inputs=inputs, outputs=out)
    return model


def prepare_dataset(dataset_path, val_size=0.2, random_state=None):
    """
    Read the data from a csv file and split the data into a train and val split.

    Args:
        dataset_path (str): Path to the dataset .csv file.
        val_size (float): Validation split size.
        random_state (int or None): Random state for train_test_split.

    Returns:
        x_train (np.ndarray): Train X data.
        x_val (np.ndarray): val X data.
        y_train (np.ndarray): Train y data.
        y_val (np.ndarray): val y data.
    """
    assert dataset_path.endswith('.csv')
    df = pd.read_csv(dataset_path, header=None)

    x = df.iloc[:, :4]
    y = df.iloc[:, 4:]

    x_train, x_val, y_train, y_val = [
        split.to_numpy() for split in train_test_split(
            x, y, test_size=val_size, random_state=random_state)]

    return x_train, x_val, y_train, y_val


def calculate_mean_and_std(array):
    """
    Calculate a mean and standard deviation of the 2D array in the batch
    dimension.
    Args:
        array (np.ndarray): 2D array to be normalized.

    Returns:
        mean (np.ndarray): Mean of the array.
        std (np.ndarray): Standard deviation of the array.

    """
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)

    return mean, std


def main(args):
    """
    Train the model on the given dataset.

    Args:
        args: Command line arguments.
    """

    args = parse_arguments(args)

    # Loading the data.
    x_train, x_val, y_train, y_val = prepare_dataset(
        args.dataset_path, val_size=args.val_size,
        random_state=args.random_state)

    # Normalizing the train data
    x_moments = calculate_mean_and_std(x_train)
    y_moments = calculate_mean_and_std(y_train)

    # Creating the model.
    model = create_fully_connected_nn((4,),
                                      x_moments,
                                      y_moments,
                                      args.model_structure)
    model.compile(optimizer=args.optimizer, loss=args.loss_func)
    model.summary()

    # Training the model.
    name = str(args.version)
    output_dir = os.path.join(args.output_dir, name)

    model.fit(
        x_train, y_train, batch_size=args.batch_size,
        epochs=args.num_epochs, validation_data=(x_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=args.early_stopping_patience),
            tf.keras.callbacks.ModelCheckpoint(output_dir,
                                               save_best_only=True)])


def parse_arguments(args):
    """
    Parse command line arguments.
    Args:
        args: Command line arguments.

    Returns:
        args: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--version', type=int,
                        required=True,
                        help='Model version.')

    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size.')

    parser.add_argument('--random_state', type=int, default=19,
                        help='Random state for train-test split.')

    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs.')

    parser.add_argument('--dataset_path', type=str, default='./dataset.csv',
                        help='Path to the dataset.')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use for training.')

    parser.add_argument('--loss_func', type=str, default='mae',
                        help='Loss function.')

    parser.add_argument('--val_size', type=int, default=0.3,
                        help='Validation split size.')

    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience.')

    parser.add_argument('--output_dir', type=str,
                        default='./checkpoints',
                        help='Output path.')

    parser.add_argument('--model_structure', type=str, default=(20, 20, 2),
                        nargs='+', help='Model structure, i. e. number of '
                                        'neurons per layer.')

    return parser.parse_args(args)


if __name__ == '__main__':
    main(sys.argv[1:])
