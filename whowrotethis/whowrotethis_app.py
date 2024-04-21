"""
Streamlit app for UI for module whowrote this

INFO: run the following command in terminal for app to open in localhost

streamlit run whowrotethis_app.py (when inside the whowrotethis model)

"""

# import libraries
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from whowrotethis import TextEmbedding, Classifier, EvaluateModel
from whowrotethis.models import EnsembledModel
from error_logger import log_error
from whowrotethis.EvaluateClassifier import evaluate


def get_wrong_pred_charts():
    df = pd.read_csv(
        f'{os.getcwd()}\\whowrotethis\\models\\model_description.csv')
    print("Model Evaluation on 10k raw text embeddings")

    # Used to draw the bar plots
    count = 0
    fig1, axs1 = plt.subplots(
        1, 5, figsize=(15, 5), tight_layout=True)
    fig2, axs2 = plt.subplots(
        1, 5, figsize=(15, 5), tight_layout=True)
    fig1.suptitle("Number of wrong predictions on 10k raw texts")
    fig2.suptitle("Number of wrong predictions on 10k preprocessed texts")

    for model in df['model_file']:
        # Test on raw unseen text embeddings
        print("Test on 10k raw text embeddings" + "-" * 20)
        report_1 = EvaluateModel(
            model, '10k_raw_unseen.csv', axs1, count)
        print(report_1)
        report_1.evaluate()
        report_1.show_wrong()

        # Test on preprocessed text embeddings
        print("Test on 10k preprocessed text embeddings" + "-" * 20)
        report_2 = EvaluateModel(
            model, '10k_preprocessed_unseen.csv', axs2, count)
        print(report_2)
        report_2.evaluate()
        report_2.show_wrong()

        count += 1

    return fig1, fig2


def get_ensemble_charts():
    data = pd.read_csv(
        f'{os.getcwd()}\\whowrotethis\\data\\10k_raw_unseen.csv')
    x_test = data.loc[:, '0': '767']
    y_test = data['label']

    # Get predictions
    model = EnsembledModel.EnsembledModel(x_test)
    y_pred_1 = model.simple_predict()
    y_pred_2 = model.weighted_predict()

    # Prepare figure
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), tight_layout=True)
    count = 0
    fig.suptitle("Number of wrong predictions using the ensembled model")
    print("Weighted Ensemble Model:")
    evaluate(y_test, y_pred_2, axs, count, "Weighted Ensembled Model")
    count += 1
    print("-" * 30)

    print("Unweighted Ensemble Model:")
    evaluate(y_test, y_pred_1, axs, count, "Unweighted Ensembled Model")

    return fig


def extract_embedding(text):
    """
    Extract word embeddings from file
    :param text: text string
    :return: embeddings
    """
    path = os.getcwd()
    # user input saved in text file
    try:
        with open(f"user_input.txt", "w", errors="ignore") as f:
            f.write(text)
    except Exception as e:
        log_error(f"{e}", "main",
                  "whowrotethis_app.py.py")
    # getting text embeddings
    embeddings = TextEmbedding(f'{path}\\user_input.txt').get_embeddings()
    return embeddings


def get_data():
    """
    Get dataframe for visualisation
    :return: models dataframe
    """
    models = pd.read_csv(f'{os.getcwd()}\\whowrotethis\\models\\model_description.csv')

    # get data
    models['accuracy'] = models['model_file'].apply(
        lambda x: EvaluateModel(x, '10k_raw_unseen.csv', 0, 0).get_acc())
    models['accuracy_raw'] = models['model_file'].apply(
        lambda x: EvaluateModel(x, '10k_preprocessed_unseen.csv', 0, 0).get_acc())
    models['classification_report'] = models['model_file'].apply(
        lambda x: EvaluateModel(x, '10k_raw_unseen.csv', 0, 0).get_report_dict())
    return models


def main():
    menu = ["Predict Text", "Model Evaluation"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Predict Text":
        st.title("Who Wrote this?")
        st.subheader("Input text")
        text = st.text_area("Input your text here")
        flag = True  # flag to check status of embeddings

        embeddings = extract_embedding(text)
        if embeddings is not None:
            predict = Classifier(embeddings)
            # getting prediction
            prediction = predict.predict_text()
        else:
            flag = False
        st.subheader("Predictions")
        if st.button('Predict', type="primary"):
            if flag:
                st.success(prediction)  # display prediction
            else:
                st.warning("Unable to extract embeddings from input text")

    elif choice == "Model Evaluation":
        st.title("Model Evaluation")
        models = get_data()
        # display data
        st.dataframe(models[['model_name', 'accuracy']])

        # display accuracy
        st.subheader("Model Accuracy ")
        st.bar_chart(models[['model_name', 'accuracy']], x='model_name', y='accuracy')

        # display classification report for each model
        for i in range(models.shape[0]):
            st.subheader(f"{models.loc[i, 'model_name']}")
            st.write(models.loc[i, 'Description'])
            report_dict = models.loc[i, 'classification_report']
            report = pd.DataFrame(report_dict)
            report = report[['0', '1']].rename(columns={'0': 'Class 0', '1': 'Class 1'})
            st.write(report)
        st.write(" ")

        # display statistics
        st.subheader("Model performance on raw vs pre-processed text")
        st.write("We consider the performance of our best model XGBoostClassifier-Xl on raw and pre-processed data")
        compare_raw = models[models['model_name'] == 'XGBoostClassifier-XL'][['accuracy', 'accuracy_raw']]
        st.dataframe(compare_raw, hide_index=True)
        st.bar_chart(compare_raw.transpose())

        st.write(
            "As seen by the bar plot above the accuracy of model trained on raw text is greater than the accuracy of the model trained on pre-processed text.")
        fig1 = get_ensemble_charts()
        fig2, fig3 = get_wrong_pred_charts()
        st.subheader("Number of wrong predictions for the ensemble models")
        st.pyplot(fig1)
        st.subheader("Number of wrong predictions in each model")
        st.pyplot(fig2)
        st.write(" ")
        st.pyplot(fig3)


if __name__ == "__main__":
    main()
