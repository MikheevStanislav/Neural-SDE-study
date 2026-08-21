from . import mujoco
from . import physionet
from . import crypto


DATASET_NAMES = ("mujoco", "physionet", "crypto")


def get_dataset(name):
    dataset = globals().get(name)
    if dataset is None or not hasattr(dataset, "get_data"):
        raise ValueError(
            "Unknown dataset {!r}; expected one of: {}.".format(
                name, ", ".join(DATASET_NAMES)
            )
        )
    return dataset


def feature_dimensions(dataset):
    fallback = getattr(dataset, "NUM_FEATURES", None)
    input_features = getattr(dataset, "NUM_INPUT_FEATURES", fallback)
    output_features = getattr(dataset, "NUM_OUTPUT_FEATURES", fallback)
    if input_features is None or output_features is None:
        raise AttributeError(
            f"Dataset {dataset.__name__} must define NUM_FEATURES or both "
            "NUM_INPUT_FEATURES and NUM_OUTPUT_FEATURES."
        )
    return int(input_features), int(output_features)
