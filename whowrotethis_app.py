import os

import streamlit as st
import pandas as pd
import random
import tensorflow as tf
from whowrotethis import TextPreprocessing, TextEmbedding, Classifier


def main():
    menu = ["Predict Text", "Model"]
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

    elif choice == "Model":
        pass


if __name__ == "__main__":
    main()
