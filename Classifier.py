"""

Module to take in the text embeddings and predict
the text as Human or AI generated
"""
import pickle


class Classifier:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def predict_text(self):
        """
        Predict the text as Human or AI generated
        :return: str
        """
        model = pickle.load(open("xgb_reg.pkl", 'rb'))
        prediction = model.predict(self.embeddings)
        print("Prediction:", prediction)
        if prediction == 1:
            return "AI Generated"
        else:
            return "Human Generated"
