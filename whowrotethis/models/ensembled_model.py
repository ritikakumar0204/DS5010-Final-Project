"""
    Here's a class to ensemble all the models using voting.
    A class for getting the performance of
    the ensembled model in EvaluateClassifier.py
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import os


class EnsembledModel:
    def __init__(self, embeddings):
        """
        :param embeddings: a dataframe of all text embeddings
        """
        self.embeddings = embeddings
        self.path = f'{os.getcwd()}\\models\\'
        self.model_names = pd.read_csv(self.path + 'model_description.csv')
        self.models, self.names = self.load_model()
        self.weights = self.set_weights()
        self.all_pred = self.get_predict()

    def load_model(self):
        models = []
        file_model = {} # key - the model; value - file name
        for model_file in self.model_names['model_file']:
            model = pickle.load(open(self.path + model_file, 'rb'))
            file_model[model] = model_file
            models.append(model)
        return models, file_model

    def get_predict(self):
        """
        Method: predict
            Predicts the labels of the given embeddings
        :return: the predictions
        """
        all_pred = pd.DataFrame()
        for i in range(len(self.models)):
            if self.names[self.models[i]] == "adaboost.pkl":
                predict_probs = self.models[i].predict(xgb.DMatrix(self.embeddings))
                predictions = (predict_probs >= 0.5).astype(int)
            else:
                predictions = self.models[i].predict(self.embeddings)

            all_pred[self.names[self.models[i]]] = predictions
            all_pred[self.names[self.models[i]] + "_weighted"] = predictions * self.weights[i]
        return all_pred

    def set_weights(self):

        weights = [0.1, 0.35, 0.1, 0.35, 0.1]

        return weights

    def simple_predict(self):
        return self.all_pred.loc[:, self.model_names['model_file']].mode(axis=1).iloc[:, 0]

    def weighted_predict(self):
        cols = [model + "_weighted" for model in self.model_names['model_file']]
        self.all_pred['sum'] = self.all_pred.loc[:, cols].sum(axis=1)
        self.all_pred['weighted_prediction'] = np.where(self.all_pred['sum'] > 0.5, 1, 0)
        return self.all_pred['weighted_prediction']








