import os

import pandas as pd
import streamlit as st
from whowrotethis import TextEmbedding, Classifier
import EvaluateModel


def main():
    menu = ["Predict Text", "Model Evaluation"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Predict Text":
        st.title("Who Wrote this?")
        st.subheader("Input text")
        text = st.text_area("Input your text here")
        path = os.getcwd()
        with open(f"user_input.txt", "w") as f:
            f.write(text)
        embeddings = TextEmbedding('user_input.txt').get_embeddings()
        predict = Classifier(embeddings)
        prediction = predict.predict_text()
        st.subheader("Predictions")
        if st.button('Predict', type="primary"):
            st.success(prediction)

    elif choice == "Model Evaluation":
        st.title("Model Evaluation")
        accuracy_dict = {}
        df = pd.read_csv(f'{os.getcwd()}\\whowrotethis\\models\\model_description.csv')
        model_list = pd.DataFrame(df)['model_file'].tolist()
        df['accuracy'] = df['model_file'].apply(lambda x: EvaluateModel.EvaluateModel(x, '10k_raw_unseen.csv').get_acc())
        df['accuracy_raw'] = df['model_file'].apply(
            lambda x: EvaluateModel.EvaluateModel(x, '10k_preprocessed_unseen.csv').get_acc())
        df['classification_report'] = df['model_file'].apply(
            lambda x: EvaluateModel.EvaluateModel(x, '10k_raw_unseen.csv').get_report_dict())
        st.dataframe(df[['model_name', 'accuracy']])
        st.subheader("Model Accuracy ")
        st.bar_chart(df[['model_name', 'accuracy']], x='model_name', y='accuracy')
        for i in range(df.shape[0]):
            st.subheader(f"{df.loc[i, 'model_name']}")
            st.write(df.loc[i, 'Description'])
            report_dict = df.loc[i, 'classification_report']
            report = pd.DataFrame(report_dict)
            report = report[['0', '1']].rename(columns={'0': 'Class 0', '1': 'Class 1'})
            st.write(report)
        st.write(" ")
        st.subheader("Model performance on raw vs pre-processed text")
        st.write("We consider the performance of our best model XGBoostClassifier-Xl on raw and pre-processed data")
        compare_raw = df[df['model_name'] == 'XGBoostClassifier-XL'][['accuracy', 'accuracy_raw']]
        st.dataframe(compare_raw, hide_index=True)
        st.bar_chart(compare_raw.transpose())

        st.write("As seen by the bar plot above the accuracy of model trained on raw text is greater than the accuracy of the model trained on pre-processed text.")


if __name__ == "__main__":
    main()
