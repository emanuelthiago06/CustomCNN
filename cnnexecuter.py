import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from model import build_model, build_model_alt
import keras
import keras_tuner

class CNNExecuter:
    def __init__(self,val_path, train_path, test_path):
        self.val_path = val_path
        self.train_path = train_path
        self.test_path = test_path

    def open_images(self):
        items = os.listdir("val_path")
        print(items)
        self.val_files_normal = []
        self.val_files_anormal = []
        for item in items:
            for file in os.listdir(os.path.join("val_path",item)):
                arquivo = os.path.join("val_path",item,file)
                if item == "0":
                    self.val_files_normal.append(arquivo)
                else:
                    self.val_files_anormal.append(arquivo)

    def pre_processing(self):
        pass # not ready

    def call_model(self, input_shape):
        model = build_model(input_shape=input_shape)
        return model
    
    def __calc_hiperparameters(self) -> None:
        tuner = keras_tuner.BayesianOptimization(
            hypermodel=build_model,
            objective="val_accuracy",
            max_trials=10,
            executions_per_trial=10,
            overwrite=True,
            directory="keras_dir",
            project_name="helloworld",
            )
        tuner.search(self.__x_train, self.__y_train, epochs=10, validation_data=(self.__x_val, self.__y_val))
        models = tuner.get_best_models(num_models=2)
        self.__best_model = models[0]
        tuner.results_summary(num_trials = 10)

    def calc_metrics(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_pred_binary = np.round(y_pred)
        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        confusion = confusion_matrix(y_test, y_pred_binary)

        print(f"Acurácia: {accuracy}")
        print(f"Precisão: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")
        print(f"Matriz de Confusão:\n{confusion}")
        auc_roc = roc_auc_score(y_test, y_pred)
        print(f"Pontuação AUC-ROC: {auc_roc}")
