from whowrotethis.models.ensembled_model import EnsembledModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt
import os

def evaluate(y_true, y_pred):
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("-" * 30)
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print("-" * 30)

    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("-" * 30)

    predictions = pd.DataFrame({
        'Actual label': y_true,
        'Predicted label': y_pred
    }, index=range(len(y_true)))

    wrong = predictions[
        predictions['Actual label'] != predictions['Predicted label']]
    values = wrong['Actual label'].astype(str).value_counts()
    print("Number of wrong labels:")
    print(values, end="\n\n")

    plt.bar(values.index, values.values)
    plt.xlabel("Actual label (0-Human, 1-AI)")
    plt.ylabel("Number of wrong predictions")
    plt.title(f"Number of wrong predictions using the ensembled model")
    plt.show()


def main():
    data = pd.read_csv(f'{os.getcwd()}\\whowrotethis\\data\\10k_raw_unseen.csv')
    x_test = data.loc[:, '0' : '767']
    y_test = data['label']
    model = EnsembledModel(x_test)
    y_pred_1 = model.simple_predict()
    y_pred_2 = model.weighted_predict()

    print("Weighted Ensemble Model:")
    evaluate(y_test, y_pred_2)
    print("-" * 30)

    print("Unweighted Ensemble Model:")
    evaluate(y_test, y_pred_1)


if __name__ == "__main__":
    main()
