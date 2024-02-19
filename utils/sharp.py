import cv2
import numpy as np


class Sharp:
    def __init__(self, diapason_white: int, low_input: int, high_input: int, gamma: float, cenny: bool):
        self.diapason_white = diapason_white
        self.high_input = high_input / 255
        self.low_input = low_input / 255
        self.gamma = 1 / gamma
        self.cenny = cenny

    def __cenny(self, image: np.ndarray) -> np.ndarray:
        image = (image * 255).astype(np.uint8)
        edges = np.clip(cv2.Canny(image, 750, 800, apertureSize=3, L2gradient=True) * -1 + 255, 0, 1)
        if self.diapason_white != -1:
            return image * edges
        return (image * edges) / 255

    def __diapason_white(self, image: np.ndarray) -> np.ndarray:
        if not self.cenny:
            image = (image * 255).astype(np.uint8)
        median_image = cv2.medianBlur(image, 3)
        _, mask2 = cv2.threshold(median_image, 255 - self.diapason_white, 255, cv2.THRESH_BINARY)
        return np.clip(image + mask2, 0, 255).astype(np.float32) / 255

    def __color_levels(self, image: np.ndarray) -> np.ndarray:
        color_levels = np.clip(((image - self.low_input) / (
                self.high_input - self.low_input)), 0.,
                               1.)
        if self.gamma != 1.0:
            color_levels = np.power(color_levels, self.gamma)
        return color_levels

    def run(self, image) -> np.ndarray:

        if self.low_input != 0 or self.high_input != 1:
            image = self.__color_levels(image)
        if self.cenny:
            image = self.__cenny(image)
        if self.diapason_white != -1:
            image = self.__diapason_white(image)
        return image
