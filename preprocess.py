import pathlib
import numpy as np
np.random.seed(0)
import pandas as pd

DATA_DIR = pathlib.Path("data")
TEST_RATIO = 0.2

def prepare_data(metadata: pd.DataFrame, label: str,
    test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get training and test metadata for binary chest x-ray classification.
    
    Args
    ----
        metadata: pd.DataFrame
            Table of metadata for each chest x-ray example
        label: str
            Positive chest x-ray label
        test_ratio: float = 0.8
            Ratio of positive labels to hold out for test set
    Returns
    -------
        tuple[pd.DataFrame, pd.DataFrame]
            Training and test set metadata
    """
    metadata = metadata.sample(frac=1.0)
    pos_data = metadata[metadata["label"] == label].reset_index(drop=True)
    neg_data = metadata[metadata["label"] == "No Finding"].reset_index(drop=True)
    
    n_pos = len(pos_data)
    assert n_pos > 0, f"Label {label} has no entries in metadata."
    
    n_test = int(test_ratio*n_pos)
    n_train = n_pos - n_test
    
    train_df = pd.concat((pos_data.loc[:n_train], neg_data.loc[:n_train]),
        axis=0, ignore_index=True)
    test_df = pd.concat((pos_data.loc[n_train:n_pos], neg_data[n_train:n_pos]),
        axis=0, ignore_index=True)

    return train_df, test_df
if __name__ == "__main__":
    # Read in metadata
    metadata = pd.read_csv(DATA_DIR / "labels.csv")
    
    train_df, test_df = prepare_data(metadata, "Cardiomegaly", test_ratio=TEST_RATIO)