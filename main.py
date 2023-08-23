
import pathlib
import pandas as pd
import tensorflow.keras as keras

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
    print("Initializing model...")
    model = get_model((IMG_HEIGHT, IMG_WIDTH, 3))
    print(model.summary())

    train_dir, test_dir = DATA_DIR / "train", DATA_DIR / "test"
    train_pos_dir, train_neg_dir = train_dir / "positive", train_dir / "negative"
    test_pos_dir, test_neg_dir = test_dir / "positive", test_dir / "negative"
    
    # dataloaders
    train_aug = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=4,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False # don't flip chest x-rays
    )
    test_aug = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    
    train_gen = train_aug.flow_from_directory(
        train_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=1,
        class_mode="binary"
    )
    test_gen = test_aug.flow_from_directory(
        test_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=1,
        class_mode="binary"
    )
    
    # Train model
    n_steps_train = len(list(train_pos_dir.iterdir())) + len(list(train_neg_dir.iterdir()))
    n_steps_test = len(list(test_pos_dir.iterdir())) + len(list(test_neg_dir.iterdir()))
    print("Training steps: ", n_steps_train)
    print("Test steps: ", n_steps_test)