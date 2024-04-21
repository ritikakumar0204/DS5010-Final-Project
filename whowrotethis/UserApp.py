"""

Python program to run streamlit app whowrotethis_app.py
"""
import os


class UserApp:
    """
    class to run streamlit app
    method:
    run_app() : opens app in localhost
    """

    def __init__(self):
        self.run_command = "streamlit run whowrotethis_app.py"

    def run_app(self):
        os.system("streamlit run whowrotethis_app.py")
