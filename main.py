from methods import *
from model import *

import cv2 as cv
import numpy as np
import glob
import streamlit as st


tmp = []

for img in sorted(glob.glob('datasets/training/images/*')):
    tmp.append(cv.imread(img, cv.IMREAD_GRAYSCALE))

vesselTrain = np.array(tmp, dtype = np.uint8)

tmp = []

for img in sorted(glob.glob('datasets/test/images/*')):
    tmp.append(cv.imread(img, cv.IMREAD_GRAYSCALE))

vesselTest = np.array(tmp, dtype = np.uint8)

tmp = []

for img in sorted(glob.glob('datasets/training/1st_manual/*')):
    tmp.append(cv.imread(img, cv.IMREAD_GRAYSCALE))

vesselManual = np.array(tmp, dtype = np.uint8)


run = CNNModel(vesselTrain, vesselManual, vesselTest)
# run.preprocess()
# run.build_model()
# run.compile_model()
# run.train_model()
# run.model.summary()
# run.visualize_predictions(1)

run.display()

