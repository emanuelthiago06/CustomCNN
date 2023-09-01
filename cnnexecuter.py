import os

class CNNExecuter:
    def __init__(self,val_path, train_path, test_path):
        self.val_path = val_path
        self.train_path = train_path
        self.test_path = test_path

    def open_images(self):
        items = os.listdir("val_path")
        print(items)
        val_files_normal = []
        val_files_anormal = []
        for item in items:
            for file in os.listdir(os.path.join("val_path",item)):
                arquivo = os.path.join("val_path",item,file)
                if item == "0":
                    val_files_normal.append(arquivo)
                else:
                    val_files_anormal.append(arquivo)