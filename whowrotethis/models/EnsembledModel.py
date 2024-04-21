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
    """
    Class: EnsembledModel
    This is a class to ensemble all the models using voting
    Methods:
    1. __init__: initialize an ensembled model with the test data
    2. load_model: load the pretrained models
    3. get_predict: returns a DataFrame
       containing the predictions of all the models on the test data
    4. simple_predict: returns the predictions of the ensembled model
       using simple voting (just compare the numbers of 0s and 1s,
       and choose the majority)
    5. weighted_predict: returns the predictions of the ensembled model
    """
    def __init__(self, embeddings):
        """
        The constructor to initialize the ensembled model
        :param embeddings: a dataframe of all text embeddings
        """

        self.embeddings = embeddings
        self.path = f'{os.getcwd()}\\whowrotethis\\models\\'
        self.model_names = pd.read_csv(self.path + 'model_description.csv')
        self.models, self.names = self.load_model()

        # Manually set the weights for the models
        # based on the accuracy got by EvaluateModel.
        self.weights = [0.1, 0.35, 0.1, 0.35, 0.1]

        self.all_pred = self.get_predict()

    def load_model(self):
        """
        Load the pretrained in the folder models
        :return: a tuple containing a list of the loaded models
            and a dictionary with the model as the key
            and the model file name as the value
        """

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
        :return: a DataFrame of all the predictions
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

    def simple_predict(self):
        """Returns the predictions of the given embeddings
        using the ensembled model with simple voting"""

        return self.all_pred.loc[:, self.model_names['model_file']].mode(axis=1).iloc[:, 0]

    def weighted_predict(self):
        """
        Returns the predictions of the given embeddings
        using the ensembled model with weighted voting"""

        cols = [model + "_weighted" for model in self.model_names['model_file']]
        self.all_pred['sum'] = self.all_pred.loc[:, cols].sum(axis=1)
        self.all_pred['weighted_prediction'] = np.where(self.all_pred['sum'] > 0.5, 1, 0)
        return self.all_pred['weighted_prediction']








