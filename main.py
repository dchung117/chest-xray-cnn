
import pathlib
import pandas as pd

from preprocess import prepare_data, create_split_dir
from model import get_model

DATA_DIR = pathlib.Path("data")
IMG_DIR = DATA_DIR / "images"

NEGATIVE_LABEL = "No Finding"
TEST_RATIO = 0.2
IMG_HEIGHT, IMG_WIDTH = 256, 256

if __name__ == "__main__":
    # Read in metadata
    metadata = pd.read_csv(DATA_DIR / "labels.csv")
    
    train_df, test_df = prepare_data(metadata, "Cardiomegaly", NEGATIVE_LABEL, test_ratio=TEST_RATIO)

    create_split_dir(train_df, True, negative_label=NEGATIVE_LABEL, data_dir=DATA_DIR)
    create_split_dir(test_df, False, negative_label=NEGATIVE_LABEL, data_dir=DATA_DIR)
    
    # Initialize model
    model = get_model((IMG_HEIGHT, IMG_WIDTH, 3))