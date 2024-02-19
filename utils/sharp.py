import cv2
import numpy as np


class Sharp:
    def __init__(self, diapason_white: int, low_input: int, high_input: int, cenny: bool):
        self.diapason_white = diapason_white / 255
        self.high_input = high_input / 255
        self.low_input = low_input / 255
        self.cenny = cenny

    @staticmethod
    def __cenny(image: np.ndarray) -> np.ndarray:
        edges = cv2.Canny((image * 255).astype(np.uint8), 750, 800, apertureSize=3, L2gradient=True)
        inverted_edges = (cv2.bitwise_not(edges)).astype(np.float32) / 255
        return image * inverted_edges

    def __diapason_white(self, image: np.ndarray) -> np.ndarray:
        median_image = cv2.medianBlur(image, 3)
        _, mask2 = cv2.threshold(median_image, 1 - self.diapason_white, 1, cv2.THRESH_BINARY)
        return np.clip(image + mask2, 0, 1)

    def __color_levels(self, image: np.ndarray) -> np.ndarray:
        return np.clip(((image - self.low_input) / (
                self.high_input - self.low_input)), 0.,
                1.)



    def run(self, image: np.ndarray) -> np.ndarray:
        if self.low_input != 0 or self.high_input != 1:
            image = self.__color_levels(image)
        if self.cenny:
            image = self.__cenny(image)
        if self.diapason_white != -1:
            image = self.__diapason_white(image)
        return image
