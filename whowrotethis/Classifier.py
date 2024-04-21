"""

Module to take in the text embeddings and predict
the text as Human or AI generated
"""
import pickle
import pandas as pd
import xgboost
import os
from error_logger import log_error


class Classifier:
    """
    Class to predict the text as Human or AI generated

    Attributes:
        embeddings: the text embeddings
        models: csv file containing model data

    Methods:
        list_models: returns a list of all the models used for evaluation
        model_description(model_name): description of the given model
        predict_text: predicts the text as Human or AI generated
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings
        try:
            self.path_curr = os.getcwd()
            self.models = pd.read_csv(f'{self.path_curr}\\whowrotethis\\models\\model_description.csv')

        except Exception as error:
            log_error(f"{error}", "__init__",
                      "Classifier.py")

    def list_models(self):
        """
        Returns a list of all the models used for evaluation
        :return: list
        """
        return self.models['model_name'].tolist()

    def model_description(self, model_name):
        """
        Prints the description of the given model
        :param model_name: name of the model
        """

        description = f'''
        Model Name: {model_name}
        Description:
        {self.models[self.models['model_name'] == model_name]['Description'].item()}
        '''
        print(description)

    def predict_text(self):
        """
        Predict the text as Human or AI generated
        :return: str
        """
        path = f'{self.path_curr}\\whowrotethis\\models\\'
        model_paths = self.models['model_file'].tolist()
        predictions = []
        for i in model_paths:
            if i == 'adaboost.pkl': # takes dmatrix as input
                self.embeddings = xgboost.DMatrix(self.embeddings)
            model = pickle.load(open(f"{path}{i}", 'rb'))
            predictions.append(model.predict(self.embeddings))

        # count the majority predictions of all the models
        count_zeros = predictions.count(0)
        count_ones = predictions.count(1)

        # majority of the prediction wins
        if count_zeros > count_ones:
            prediction = 0
        else:
            prediction = 1

        if prediction == 1:
            return 'AI generated'
        elif prediction == 0:
            return 'Human generated'
