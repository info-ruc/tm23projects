# -*- coding: utf-8 -*-

"""

"""

import tensorflow.keras as keras
import numpy as np
from sklearn import metrics
import os

from preprocess import preprocesser
from config import Config
from model import TextCNN,LSTM


if __name__ == '__main__':
    # CNN_model = TextCNN()
    # CNN_model.train(3)
    # CNN_model.test()

    CNN_model = LSTM()
    CNN_model.train(3)
    CNN_model.test()
