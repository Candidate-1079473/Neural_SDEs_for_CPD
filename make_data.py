"""Create real-world univariate and multivariate time series containing a change-point"""
import os
from pyts.datasets import fetch_ucr_dataset, fetch_uea_dataset
import numpy as np


def _load_ucr_uea_data(dataset_name):
    """Load UCR and UEA datasets from http://www.timeseriesclassification.com/dataset.php"""
    try:
        data_train, data_test, target_train, target_test = fetch_ucr_dataset(
            dataset_name, return_X_y=True
        )
    except ValueError:
        data_train, data_test, target_train, target_test = fetch_uea_dataset(
            dataset_name, return_X_y=True
        )
    data, target = np.concatenate([data_train, data_test]), np.concatenate(
        [target_train, target_test]
    )
    if data.shape[1] > 750:
        data = data[:, :: data.shape[1] // 750]

    return np.nan_to_num(data), target


def make_univariate_timeseries(directory, dataset_name, batch_size):
    """Save `batch_size` univariate time series in `directory`.
    * Each time series contains one change-point.
    * A change-point marks the change between two randomly chosen classes."""

    data, target = _load_ucr_uea_data(dataset_name)

    classes = set(target)
    for i in range(batch_size):
        # randomly select classes between which the time series changes
        class_A, class_B = np.random.choice(list(classes), size=(2,), replace=False)
        class_A, class_B = np.random.permutation([class_A, class_B])

        data_A = data[target == class_A]
        data_B = data[target == class_B]

        # determine how much of each class the time series contains
        max_reps = min(data_A.shape[0], data_B.shape[0], 25)
        reps_A = np.random.randint(10, max_reps)
        reps_B = np.random.randint(5, reps_A)

        change_point = reps_A * data_A.shape[1]

        # remove the periodicity to make change-point detection harder
        data_A = data_A[:reps_A].reshape((change_point, -1))
        data_B = data_B[:reps_B].reshape((reps_B * data_B.shape[1], -1))
        data_AB = np.concatenate([data_A, data_B], axis=0)

        assert change_point >= len(data_AB) // 2

        # save time series
        np.save(
            os.path.join(
                directory,
                f"{dataset_name}_{i}_change_from_class_{class_A}_to_class_{class_B}_at_{change_point:06d}.npy",
            ),
            data_AB,
        )


def make_multivariate_timeseries(directory, dataset_name, batch_size):
    """
    Save `batch_size` multivariate time series in `directory`.

    * Each time series contains one change-point.
    * A change-point marks the change between two randomly chosen classes.
    """
    name_to_components = {
        "GestureMidAir": ["GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3"],
        "Cricket": ["CricketX", "CricketZ", "CricketY"],
        "NonInvasiveFetalECGThorax": [
            "NonInvasiveFetalECGThorax1",
            "NonInvasiveFetalECGThorax2",
        ],
    }
    component_names = name_to_components[dataset_name]
    DATA_TARGET = [
        (component_name, _load_ucr_uea_data(component_name))
        for component_name in component_names
    ]

    _, (_, target) = DATA_TARGET[0]
    classes = set(target)
    for i in range(batch_size):
        # randomly select classes between which the time series changes
        class_A, class_B = np.random.choice(list(classes), size=(2,), replace=False)
        class_A, class_B = np.random.permutation([class_A, class_B])
        # make multivariate time series from univariate ones
        data_A = np.stack(
            [
                data[target == class_A]
                for (component_name, (data, target)) in DATA_TARGET
            ],
            axis=-1,
        )
        data_B = np.stack(
            [
                data[target == class_B]
                for (component_name, (data, target)) in DATA_TARGET
            ],
            axis=-1,
        )

        # determine how much of each class the time series contains
        max_reps = min(data_A.shape[0], data_B.shape[0], 25)
        reps_A = np.random.randint(10, max_reps)
        reps_B = np.random.randint(5, reps_A)

        change_point = reps_A * data_A.shape[1]

        # remove the periodicity to make change-point detection harder
        data_A = data_A[:reps_A].reshape((change_point, -1))
        data_B = data_B[:reps_B].reshape((reps_B * data_B.shape[1], -1))
        data_AB = np.concatenate([data_A, data_B], axis=0)

        assert change_point >= len(data_AB) // 2

        np.save(
            os.path.join(
                directory,
                f"{dataset_name}_{i}_change_from_class_{class_A}_to_class_{class_B}_at_{change_point:06d}.npy",
            ),
            data_AB,
        )


def make_data(directory, dataset_names, batch_size):
    """For each dataset name, save `batch_size` time series containing one change-point into `directory`"""
    for dataset_name in dataset_names:
        if dataset_name in ["GestureMidAir", "Cricket", "NonInvasiveFetalECGThorax"]:
            make_multivariate_timeseries(directory, dataset_name, batch_size)
        else:
            make_univariate_timeseries(directory, dataset_name, batch_size)