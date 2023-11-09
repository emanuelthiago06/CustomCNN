import sys
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
from skimage.filters import (threshold_otsu,sobel, threshold_triangle, threshold_niblack, threshold_sauvola)
from skimage import img_as_ubyte

class PreProcessing:
    def __init__(self, **kwargs):
        self.files = [] if "files" not in kwargs else kwargs["files"]
        self.images = []

    def read_img(self, janelamento = False):
        for file in self.files:
            img_dc = pydicom.dcmread(file)
            if janelamento:
                img_dc = self.make_jan(img_dc, janelamento=janelamento)
            self.images.append(img_dc)

    @staticmethod
    def make_jan(image, janelamento: list):
        """Janelamento : Lista com o valor mínimo e máximo de janela, transforma esse valor em um array numpy"""
        min_valor = min(janelamento)
        max_valor = max(janelamento)
        imagem_janelada = np.copy(image)
        imagem_janelada[image < min_valor] = min_valor
        imagem_janelada[image > max_valor] = max_valor
        return imagem_janelada
    
    @staticmethod
    def apply_otsu(image):
        thresh_otsu = threshold_otsu(image)
        binary_img_1 = image > thresh_otsu
        binary_img_1 = img_as_ubyte(binary_img_1)
        return binary_img_1
