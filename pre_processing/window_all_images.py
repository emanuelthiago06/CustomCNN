import os
from preprocessing import PreProcessing
import cv2
import pydicom
import matplotlib.pyplot as plt

NEW_PATH = "/home/emanuel/Documents/Testes so slices patologicos avci/dataset_processed"
OLD_BASE_PATH = "/home/emanuel/Documents/Testes so slices patologicos avci/dataset_no_preprocess"

def pre_process_dataset(current_path: str, classific: int, dataset_type: str):
    preprocess = PreProcessing()
    items = os.listdir(current_path)
    for item in items:
        for file in os.listdir(os.path.join(current_path,item)):
            arquivo = os.path.join(current_path,item,file)
            ds = pydicom.dcmread(arquivo)
            image = ds.pixel_array
            img_jan = preprocess.make_jan(image = image, janelamento=[1000,1150])
            # cv2.imwrite(filename=os.path.join(NEW_PATH,classific,dataset_type),img=img_jan)
            # cv2.imshow("jan", img_jan)
            plt.imshow(img_jan, cmap="gray")
            plt.title('Windowed DICOM Image')
            plt.axis('off') 
            plt.savefig(os.path.join(NEW_PATH,classific,dataset_type,file[:-4]+".png"), bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
             
if __name__ == "__main__":
    for i in ["0","1"]:
        for j in ["val","test","train"]:
            current_path = os.path.join(OLD_BASE_PATH,i,j)
            pre_process_dataset(current_path=current_path, classific=i, dataset_type=j)