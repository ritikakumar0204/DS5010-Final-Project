"""
    Here's a class EvaluateModel to evaluate all the models we've got.
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score)


class EvaluateModel:
    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name

        self.model_file = '../models/' + model_name
        self.data_file = '../data/' + data_name

        self.model = self.load_model()

        self.y_test, self.x_test = self.load_data()

        if self.model_name == 'adaboost.pkl':
            self.y_pred_prob = self.model.predict(xgb.DMatrix(self.x_test))
            self.y_pred = (self.y_pred_prob >= 0.5).astype(int)
        else:
            self.y_pred = self.model.predict(self.x_test)
            self.y_pred_prob = self.model.predict_proba(self.x_test)[:, 1]

    def load_model(self):
        with open(self.model_file, 'rb') as f:
            model = pickle.load(f)

        return model

    def load_data(self):
        data = pd.read_csv(self.data_file)
        return data['label'], data.loc[:, '0': '767']

    def get_acc(self):
        return accuracy_score(self.y_test, self.y_pred)

    def get_report(self):
        return classification_report(self.y_test, self.y_pred)

    def get_roc_auc(self):
        return roc_auc_score(self.y_test, self.y_pred_prob)

    def evaluate(self):
        print(f"Accuracy: {self.get_acc()}")
        print("-" * 30)

        print("Classification Report:")
        print(self.get_report())
        print("-" * 30)

        print(f"ROC AUC Score: {self.get_roc_auc()}")
        print("-" * 30)

    def get_wrong(self):
        predictions = pd.DataFrame({
            'Actual label': self.y_test,
            'Predicted label': self.y_pred
        }, index=range(len(self.y_test)))

        return predictions[predictions['Actual label']
                           != predictions['Predicted label']]

    def show_wrong(self):
        wrong = self.get_wrong()
        values = wrong['Actual label'].astype(str).value_counts()
        print("Number of wrong labels:")
        print(values)

        plt.bar(values.index, values.values)
        plt.xlabel("Actual label (0-Human, 1-AI)")
        plt.ylabel("Number of wrong predictions")
        plt.title(f"Number of wrong predictions using model: {self.model_name}")
        plt.show()

    def __str__(self):
        return f"Model: {self.model_name}, Data: {self.data_name}"


def main():
    df = pd.read_csv('../models/model_description.csv')
    print("Model Evaluation on 10k raw text embeddings")
    for model in df['model_file']:
        report = EvaluateModel(model, '10k_raw_unseen.csv')
        report.evaluate()
        report.show_wrong()


if __name__ == '__main__':
    main()