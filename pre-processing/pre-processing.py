import sys
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
from skimage.filters import (threshold_otsu,sobel, threshold_triangle, threshold_niblack, threshold_sauvola)
from skimage import img_as_ubyte

class PreProcessing:
    def __init__(self, **kwargs):
        self.files = kwargs["files"]
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
        janelamento_array = np.arange(janelamento[0],janelamento[1])
        min_valor = min(janelamento_array)
        max_valor = max(janelamento_array)
        imagem_janelada = np.copy(image)
        imagem_janelada[image < min_valor] = min_valor
        imagem_janelada[image > max_valor] = max_valor
        return imagem_janelada

