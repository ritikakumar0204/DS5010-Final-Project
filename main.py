"""
Program to demonstrate usage of whowrotethis
python module
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from whowrotethis import (TextPreprocessing, TextEmbedding, Classifier,
                          EvaluateModel)
from whowrotethis.models.EnsembledModel import EnsembledModel
from whowrotethis.EvaluateClassifier import evaluate


def main():
    text = "This is a test."
    processor = TextPreprocessing(text, file_given=False)
    preprocessed_text = processor.preprocess()
    print(preprocessed_text)

    embeddings = TextEmbedding('text.txt').get_embeddings()
    predict = Classifier(embeddings)
    print(predict.predict_text())

    # # Evaluate the ensembled models------------------------------------------
    # # Load data
    # data = pd.read_csv(
    #     f'{os.getcwd()}\\whowrotethis\\data\\10k_raw_unseen.csv')
    # x_test = data.loc[:, '0' : '767']
    # y_test = data['label']
    #
    # # Get predictions
    # model = EnsembledModel(x_test)
    # y_pred_1 = model.simple_predict()
    # y_pred_2 = model.weighted_predict()
    #
    # # Prepare figure
    # fig, axs = plt.subplots(1, 2, figsize=(15, 5), tight_layout=True)
    # count = 0
    # fig.suptitle("Number of wrong predictions using the ensembled model")
    #
    # # Evaluate models
    # print("Weighted Ensemble Model:")
    # evaluate(y_test, y_pred_2, axs, count, "Weighted Ensembled Model")
    # count += 1
    # print("-" * 30)
    #
    # print("Unweighted Ensemble Model:")
    # evaluate(y_test, y_pred_1, axs, count, "Unweighted Ensembled Model")
    #
    # plt.show()
    # # -----------------------------------------------------------------------

    # # Evaluate all the models separately-------------------------------------
    # df = pd.read_csv(
    #     f'{os.getcwd()}\\whowrotethis\\models\\model_description.csv')
    # print("Model Evaluation on 10k raw text embeddings")
    #
    # # Used to draw the bar plots
    # count = 0
    # fig1, axs1 = plt.subplots(
    #     1, 5, figsize=(15, 5), tight_layout=True)
    # fig2, axs2 = plt.subplots(
    #     1, 5, figsize=(15, 5), tight_layout=True)
    # fig1.suptitle("Number of wrong predictions on 10k raw texts")
    # fig2.suptitle("Number of wrong predictions on 10k preprocessed texts")
    #
    # for model in df['model_file']:
    #     # Test on raw unseen text embeddings
    #     print("Test on 10k raw text embeddings" + "-" * 20)
    #     report_1 = EvaluateModel(
    #         model, '10k_raw_unseen.csv', axs1, count)
    #     print(report_1)
    #     report_1.evaluate()
    #     report_1.show_wrong()
    #
    #     # Test on preprocessed text embeddings
    #     print("Test on 10k preprocessed text embeddings" + "-" * 20)
    #     report_2 = EvaluateModel(
    #         model, '10k_preprocessed_unseen.csv', axs2, count)
    #     print(report_2)
    #     report_2.evaluate()
    #     report_2.show_wrong()
    #
    #     count += 1
    #
    # plt.show()
    # # -----------------------------------------------------------------------


if __name__ == "__main__":
    main()
