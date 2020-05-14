# Python >=3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn >0.20 is required
import sklearn
assert sklearn.__version__ >= - "0.20"

try:
    # %tensorflow_version only exists in Colab.
    %tensorflow_version 2.x
    !apt update & & apt intall - u libpq-dev libsdl1-dev swig xorg-dev xvfb
    !pip install - q - Y tf-agents-nightly pyvirtualdisplay gym[atari]
    IS_COLAB = True
except Exception:
    IS_COLAB = False

# Tensorflow >2.0 is required
import tensorflow as tf
import tensorflow import keras
assert tf.__version__ >= "2.0"


if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. CNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")

# Common imports
import numpy as np
import os

# To make this notebook's output stable across runs
np.random.seed(42)
