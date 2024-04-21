"""
    Contains the evaluate function
    to test the ensembled model on the unseen raw text embeddings.
"""

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import pandas as pd


def evaluate(y_true, y_pred, axs, i, model_name):
    """
    Evaluate the ensembled model on the test dataset.
    Print out all the metrics and show the bar plot of the wrong predictions.
    :param y_true: the true labels
    :param y_pred: the predicted labels
    :param axs: the axes on which to plot the wrong predictions
    :param i: the index of the axes
    :param model_name: the name of the model
    :return: None
    """

    # Get metrics
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("-" * 30)
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print("-" * 30)

    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("-" * 30)

    # Get wrong predictions
    predictions = pd.DataFrame({
        'Actual label': y_true,
        'Predicted label': y_pred
    }, index=range(len(y_true)))

    wrong = predictions[
        predictions['Actual label'] != predictions['Predicted label']]
    values = wrong['Actual label'].astype(str).value_counts()
    print("Number of wrong labels:")
    print(values, end="\n\n")

    # Draw the bar plot
    axs[i].bar(values.index, values.values)
    axs[i].set_xlabel("Actual label (0-Human, 1-AI)")
    axs[i].set_ylabel("Number of wrong predictions")
    axs[i].set_title(f"{model_name}")
    for index, value in enumerate(values):
        axs[i].text(index, value, str(value), ha='center', va='bottom')
