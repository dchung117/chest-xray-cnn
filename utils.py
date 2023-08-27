from PIL import Image
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def load_image_as_array(img_file: str, img_height: int,
    img_width: int) -> np.ndarray:
    """
    Reads and converts RGB chest x-ray image into numpy array.
    
    Args
    ----
        img_file:
            Path to chest x-ray image
        img_height: int
            Resized image height
        img_width: int
            Resized image width

    Returns
    -------
        np.ndarray
            Chest x-ray image in numpy array format
    """
    img = Image.open(img_file).resize((img_width, img_height)).convert("RGB")
    return np.array(img).reshape((img_height, img_width, 3)).astype(np.uint8)

def get_metadata_sample(metadata: pd.DataFrame, label: str,
    n: int = 6) -> pd.DataFrame:
    """
    Sample from metadata for a specific chest x-ray label
    
    Args
    ----
        metadata: pd.DataFrame
            Table of metadata for each chest x-ray example
        label: str
            Chest x-ray label
        n: int = 6
            Sample size
    
    Returns
    -------
        pd.DataFrame
            Dataframe sample of the label
    """
    sample = metadata[metadata["label"] == label].reset_index(drop=True)
    return sample.loc[:n]

def plot_images(imgs: list[np.ndarray], label: str) -> None:
    """
    Plot chest x-ray images of given label.
    
    Args
    ----
        imgs: list[np.ndarray]
            List of images in numpy array format
        label: str
            Chest x-ray label for images

    Returns
    -------
        None
    """
    fig, ax = plt.subplots(2, 3)
    for i in range(6):
        row_idx = i // 3
        col_idx = i % 3
        ax[row_idx, col_idx].imshow(imgs[i])
        ax[row_idx, col_idx].set_xticks([])
        ax[row_idx, col_idx].set_yticks([])
    plt.suptitle(label)
    plt.show()

def predict_image(filename: pathlib.Path, model: tf.keras.Model,
    img_height: int, img_width: int, thresh: float = 0.5) -> tuple[float, str]:
    """
    Get chest x-ray prediction for a given image.

    Args
    ----
        filename: pathlib.Path
            Chest x-ray image file
        model: tf.keras.Model
            Trained image classification model
        img_height: int
            Resized image height
        img_width: int
            Resized image width
        thresh: float = 0.5
            Threshold for positive prediction
    
    Returns
    -------
        tuple[float, str]
            Logit of positive class prediction and predicted class
    """
    img_array = np.expand_dims(
        load_image_as_array(filename, img_height, img_width) / 255.0,
        axis=0)

    logit = model.predict(img_array)[0][0]
    if logit > thresh:
        return logit, "pos"
    else:
        return logit, "neg"

def plot_prediction(filename: pathlib.Path, label: str, pred: str,
    confidence: float, img_height: int, img_width: int) -> None:
    """
    Plot chest x-ray image w/ ground truth label, prediction, and confidence.
    
    Args
    ----
        filename: pathlib.Path
            Path to chest x-ray image
        label: str
            Ground truth label
        pred: str
            Model prediction
        confidence: float
            Model confidence in positive class
        img_height: int
            Resized image height
        img_width: int
            Resized image width

    Returns
    -------
    None
    """
    img_array = load_image_as_array(filename, img_height, img_width)
    plt.imshow(img_array)
    title = f"Label: {label} Pred: {pred} Confidence: {confidence:.4f}"
    plt.title(title)
    plt.axis("off")
    plt.show()