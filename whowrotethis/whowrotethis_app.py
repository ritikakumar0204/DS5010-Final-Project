"""
Streamlit app for UI for module whowrote this

INFO: run the following command in terminal for app to open in localhost

streamlit run whowrotethis_app.py

"""

# import libraries
import os
import pandas as pd
import streamlit as st
from whowrotethis import TextEmbedding, Classifier, TextPreprocessing
import EvaluateModel


def main():
    menu = ["Predict Text", "Model Evaluation"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Predict Text":
        st.title("Who Wrote this?")
        st.subheader("Input text")
        text = st.text_area("Input your text here")

        # file user_input.txt saves the data that user inputs
        path = os.getcwd()
        preprocessed = TextPreprocessing(text, file_given=False).preprocess()
        with open(f"user_input.txt", "w", errors="ignore") as f:
            f.write(text)
        # getting text embeddings
        embeddings = TextEmbedding(f'{path}\\user_input.txt').get_embeddings()
        predict = Classifier(embeddings)

        # getting prediction
        prediction = predict.predict_text()
        st.subheader("Predictions")
        if st.button('Predict', type="primary"):
            st.success(prediction)  # display prediction

    elif choice == "Model Evaluation":
        st.title("Model Evaluation")
        models = pd.read_csv(f'{os.getcwd()}\\whowrotethis\\models\\model_description.csv')

        # get data
        models['accuracy'] = models['model_file'].apply(
            lambda x: EvaluateModel.EvaluateModel(x, '10k_raw_unseen.csv').get_acc())
        models['accuracy_raw'] = models['model_file'].apply(
            lambda x: EvaluateModel.EvaluateModel(x, '10k_preprocessed_unseen.csv').get_acc())
        models['classification_report'] = models['model_file'].apply(
            lambda x: EvaluateModel.EvaluateModel(x, '10k_raw_unseen.csv').get_report_dict())

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


if __name__ == "__main__":
    main()
