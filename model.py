import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import streamlit as st
from keras.src.layers import BatchNormalization
from tensorflow.python.keras.saving.save import load_model

from methods import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, saving
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class CNNModel:

    def __init__(self, train_set, labels_set, test_set):
        self.X_train_set = train_set
        self.Y_labels_set = labels_set
        self.test_set = test_set
        if os.path.exists('/Users/jakubkubiak47/PycharmProjects/MachineLearning/models/test.keras'):
            self.model = saving.load_model("/Users/jakubkubiak47/PycharmProjects/MachineLearning/models/test.keras")
        else:
            self.model = None

    def display(self):
        st.title("Image Segmentation with CNN")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg","tif","tiff"])

        if uploaded_file is not None:

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            input_image = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)

            preprocessed_image = self.preprocess_image_for_model(input_image)

            st.image(input_image, caption='Uploaded Image', use_column_width=False)

            if self.model is None:
                st.warning("Model not loaded.")
                return

            prediction = self.model.predict(preprocessed_image)
            prediction = (prediction.squeeze() * 255).astype(np.uint8)

            prediction = cv.createCLAHE(clipLimit=20.0, tileGridSize=(50, 50)).apply(prediction)

            st.image(prediction, caption='Predicted Mask', use_column_width=False)

    def preprocess_image_for_model(self, image):

        square_image = ImgProc.to_square(self, image)
        image_resized = cv.resize(square_image, (256, 256), interpolation=cv.INTER_CUBIC)
        expanded_dims = np.expand_dims(image_resized, axis=(0, -1))
        image = expanded_dims / 255.0

        return image


    def preprocess(self):
        processed_X = []
        processed_Y = []
        processed_test = []

        for test_img in self.test_set:
            test_img = ImgProc.to_square(self, test_img)
            test_img = cv.resize(test_img, (256, 256), interpolation=cv.INTER_CUBIC)
            test_img = np.expand_dims(test_img, axis=(0,-1))
            # test_img = np.asarray(test_img)[None,...]
            processed_test.append(test_img)

        self.test_set = np.array(processed_test)
        self.test_set = self.test_set / 255.0


        for og_img, label_img in zip(self.X_train_set, self.Y_labels_set):

            og_img, label_img = ImgProc.to_square(self, og_img), ImgProc.to_square(self, label_img)
            og_img, label_img = (cv.resize(og_img, (256, 256), interpolation=cv.INTER_CUBIC),
                                 cv.resize(label_img, (256, 256), interpolation=cv.INTER_CUBIC))
            og_img = cv.createCLAHE(clipLimit= 7.0, tileGridSize=(20,20)).apply(og_img)

            processed_X.append(og_img)
            processed_Y.append(label_img)

            angle = random.uniform(-45, 45)
            trans_test1 = cv.warpAffine(og_img, cv.getRotationMatrix2D((128,128), angle, 1.0), (256, 256))
            trans_label1 = cv.warpAffine(label_img, cv.getRotationMatrix2D((128,128), angle, 1.0), (256, 256))

            processed_X.append(trans_test1)
            processed_Y.append(trans_label1)

            x_start = random.randint(0, 112)
            y_start = random.randint(0, 112)

            cropped_og_img = og_img[y_start:y_start + 100, x_start:x_start + 100]
            cropped_label_img = label_img[y_start:y_start + 100, x_start:x_start + 100]
            trans_test2 = cv.resize(cropped_og_img, (256,256), interpolation=cv.INTER_CUBIC)
            trans_label2 = cv.resize(cropped_label_img, (256,256), interpolation=cv.INTER_CUBIC)

            processed_X.append(trans_test2)
            processed_Y.append(trans_label2)


            flip_type = random.choice([-1, 0, 1])
            trans_test3 = cv.flip(og_img, flip_type)
            trans_label3 = cv.flip(label_img, flip_type)

            processed_X.append(trans_test3)
            processed_Y.append(trans_label3)


            trans_test4 = cv.GaussianBlur(og_img, (3, 3), 0)

            processed_X.append(trans_test4)
            processed_Y.append(label_img)

        self.X_train_set = np.array(processed_X)
        self.Y_labels_set = np.array(processed_Y)

        self.X_train_set = self.X_train_set / 255.0
        self.Y_labels_set = self.Y_labels_set / 255.0


    def build_model(self):

        size = self.X_train_set[0].shape

        inputs = layers.Input(shape=(size[0], size[1], 1))

        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        p1 = BatchNormalization()(p1)

        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        p2 = BatchNormalization()(p2)

        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
        c3 = BatchNormalization()(c3)

        u1 = layers.UpSampling2D((2, 2))(c3)
        u1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
        u1 = BatchNormalization()(u1)
        u1 = layers.Concatenate()([u1, c2])

        u2 = layers.UpSampling2D((2, 2))(u1)
        u2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
        u2 = BatchNormalization()(u2)
        u2 = layers.Concatenate()([u2, c1])

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u2)

        self.model = models.Model(inputs=[inputs], outputs=[outputs])



    def compile_model(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    def train_model(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        checkpoint = ModelCheckpoint('unet_best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.0001, verbose=1)

        self.model.fit(self.X_train_set, self.Y_labels_set, epochs = 10, batch_size =8, callbacks=[early_stopping, checkpoint, reduce_lr])

        self.model.save('/Users/jakubkubiak47/PycharmProjects/MachineLearning/models/test.keras')

    def test_model(self):
        test_accuracy = self.model.evaluate()
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


    def visualize_predictions(self, num_examples=1):
        for i in range(num_examples):
            prediction = self.model.predict(self.test_set[i])

            prediction_rescaled = (prediction.squeeze() * 255).astype(np.uint8)

            clahe = cv.createCLAHE(clipLimit=20.0, tileGridSize=(50, 50))
            prediction_clahe = clahe.apply(prediction_rescaled)


            # thresholded = cv.morphologyEx(thresholded, cv.MORPH_OPEN,(3,3))
            vesselness = prediction_clahe


            # Display results
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(self.test_set[i].squeeze(), cmap='gray')
            plt.title('Original Image')

            plt.subplot(1, 2, 2)
            plt.imshow(vesselness.squeeze(), cmap='gray')
            plt.title('Prediction')

            plt.show()

