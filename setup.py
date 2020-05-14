# Python >=3.5 is required
from tensorflow import keras
import tensorflow as tf
import sklearn
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn >0.20 is required
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
assert tf.__version__ >= "2.0"
