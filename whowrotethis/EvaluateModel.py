"""
    Here's a class EvaluateModel to evaluate all the models we've got.
"""

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)


class EvaluateModel:
    """
    Class: EvaluateModel
        This is a class to evaluate all the models separately

    Methods:
    1. __init__: the constructor
    2. load_model: load the model in the folder named models
    3. load_data: load the unseen text embeddings in the folder named data
    4. get_report_dict: returns the classification report
       in a dictionary format used in streamlit
    5. get_matrix: returns the confusion matrix
    6. get_acc: returns the accuracy of the model
    7. get_report: returns the classification report
    8. get_roc_auc: returns the roc auc score of the model
    9. evaluate: combines all the metrics together and print out the results
    10. get_wrong: returns a DataFrame that
        contains the wrong predictions (False negatives and False positives)
    11. show_wrong: print out the value_counts of the wrong predictions and
        plot it out in a histogram.
    12. __str___: defines how the object is printed
    """

    def __init__(self, model_name, data_name, axs, i):
        """
        The constructor of the EvaluateModel class
        :param model_name: the model's file name
        :param data_name: the unseen text embedding's file name
        :param axs: the axs to draw the bar plot
        :param i: the index of the axs to draw the bar plot
        """

        self.model_name = model_name
        self.data_name = data_name
        self.axs = axs
        self.i = i

        # Get file directories
        self.path = f"{os.getcwd()}\\whowrotethis\\"
        self.model_file = self.path + 'models\\' + model_name
        self.data_file = self.path + 'data\\' + data_name

        # Load model and data
        self.model = self.load_model()
        self.y_test, self.x_test = self.load_data()

        # Get predictions
        if self.model_name == 'adaboost.pkl':
            self.y_pred_prob = self.model.predict(xgb.DMatrix(self.x_test))
            self.y_pred = (self.y_pred_prob >= 0.5).astype(int)
        else:
            self.y_pred = self.model.predict(self.x_test)
            self.y_pred_prob = self.model.predict_proba(self.x_test)[:, 1]

    def load_model(self):
        """
        Loads the trained model in folder models
        :return: the model
        """
        with open(self.model_file, 'rb') as f:
            model = pickle.load(f)

        return model

    def load_data(self):
        """
        Loads the unseen text embeddings in folder data
        :return: a tuple containing
            the targets (Series) and the unseen text embeddings (DataFrame)
        """
        data = pd.read_csv(self.data_file)
        return data['label'], data.loc[:, '0': '767']

    def get_report_dict(self):
        """
        Returns the classification report in a dictionary format.
        """
        return classification_report(self.y_test, self.y_pred, output_dict=True)

    def get_matrix(self):
        """
        Returns the confusion matrix.
        """
        return confusion_matrix(self.y_test, self.y_pred)

    def get_acc(self):
        """
        Returns the accuracy of the model on the test dataset.
        """
        return accuracy_score(self.y_test, self.y_pred)

    def get_report(self):
        """
        Returns the classification report.
        """
        return classification_report(self.y_test, self.y_pred)

    def get_roc_auc(self):
        """
        Returns the ROC AUC of the model on the test dataset.
        """
        return roc_auc_score(self.y_test, self.y_pred_prob)

    def evaluate(self):
        """
        Evaluates the model on the test dataset
        Print out all metrics
        """

        print("Confusion Matrix:")
        print(self.get_matrix())
        print("-" * 30)

        print(f"Accuracy: {self.get_acc()}")
        print("-" * 30)

        print("Classification Report:")
        print(self.get_report())
        print("-" * 30)

        print(f"ROC AUC Score: {self.get_roc_auc()}")
        print("-" * 30)

    def get_wrong(self):
        """
        Returns a DataFrame containing the wrong predictions
        """

        # Put y_test and y_pred into a DataFrame
        predictions = pd.DataFrame({
            'Actual label': self.y_test,
            'Predicted label': self.y_pred
        }, index=range(len(self.y_test)))

        # Returns all rows that differ in the two columns
        return predictions[predictions['Actual label']
                           != predictions['Predicted label']]

    def show_wrong(self):
        """
        Count the number of wrong predictions when the actual label is 0
        and the number of wrong predictions when the actual label is 1.
        And plot the two numbers into a bar plot. Returns None.
        """
        wrong = self.get_wrong()
        values = wrong['Actual label'].astype(str).value_counts()
        print("Number of wrong labels:")
        print(values)

        self.axs[self.i].bar(values.index, values.values)
        self.axs[self.i].set_xlabel("Actual label (0-Human, 1-AI)")
        self.axs[self.i].set_ylabel("Number of wrong predictions")
        self.axs[self.i].set_title(f"Model: {self.model_name}")
        for index, value in enumerate(values):
            self.axs[self.i].text(index, value, str(value), ha='center', va='bottom')

    def __str__(self):
        """
        String representation of the EvaluateModel object
        """
        return f"Model: {self.model_name}, Data: {self.data_name}"


def main():
    df = pd.read_csv(f'{os.getcwd()}\\models\\model_description.csv')
    print("Model Evaluation on 10k raw text embeddings")

    # Used to draw the bar plots
    count = 0
    fig1, axs1 = plt.subplots(1, 5, figsize=(15, 5), tight_layout=True)
    fig2, axs2 = plt.subplots(1, 5, figsize=(15, 5), tight_layout=True)
    fig1.suptitle("Number of wrong predictions on 10k raw texts")
    fig2.suptitle("Number of wrong predictions on 10k preprocessed texts")

    for model in df['model_file']:
        # Test on raw unseen text embeddings
        print("Test on 10k raw text embeddings" + "-" * 20)
        report_1 = EvaluateModel(model, '10k_raw_unseen.csv', axs1, count)
        print(report_1)
        report_1.evaluate()
        report_1.show_wrong()

        # Test on preprocessed text embeddings
        print("Test on 10k preprocessed text embeddings" + "-" * 20)
        report_2 = EvaluateModel(model, '10k_preprocessed_unseen.csv', axs2, count)
        print(report_2)
        report_2.evaluate()
        report_2.show_wrong()

        count += 1

    plt.show()


if __name__ == '__main__':
    main()