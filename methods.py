import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


class ImgProc:
    def __init__(self, images):
        self.outputImages = None
        self.images = images


    def show_first(self):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        axs[0].imshow(self.images[0], cmap='gray')
        axs[0].set_title("Original Image")
        axs[1].imshow(self.outputImages[0], cmap='gray')
        axs[1].set_title("Processed Image")

        plt.show()

    def show(self):

        for old, new in zip(self.images, self.outputImages):

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

            axs[0].imshow(old, cmap='gray')
            axs[0].set_title("Original Image")
            axs[1].imshow(new, cmap='gray')
            axs[1].set_title("Processed Image")

            plt.show()

    @staticmethod
    def to_square(self, image):
        h, w = image.shape[:2]
        a = min(h, w)
        delta_h = abs(a - h) // 2
        delta_w = abs(a - w) // 2
        new_image = np.zeros((a, a), dtype=image.dtype)
        new_image = image[delta_h:h-delta_h, delta_w:w-delta_w]
        return new_image


    def basicOperations(self):

        self.outputImages = []

        for img in self.images:
            img = cv.resize(img, (256,256))
            img = cv.createCLAHE(clipLimit= 7.0, tileGridSize=(20,20)).apply(img)
            img = cv.bilateralFilter(img, 6,25,2)
            img = cv.adaptiveThreshold(img,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, 10)
            img = cv.morphologyEx(img, cv.MORPH_CLOSE, np.ones((2,2), np.uint8))

            self.outputImages.append(img)