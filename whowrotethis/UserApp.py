"""
A Python module for launching a Streamlit application.

Using UserApp class to start the Streamlit application.
The class is designed to execute a shell command that launches the application on localhost.

"""
import os


class UserApp:
    """
    Class to run Streamlit app
    method:
    run_app(): Executes the command stored in self.run_command to start the application and
    opens app in localhost.
    """

    def __init__(self):
        self.run_command = "streamlit run whowrotethis_app.py"

    def run_app(self):
        """
            Launches the Streamlit application

            This method uses the os.system function to call the command line and execute the Streamlit run command.
            It effectively starts the server and hosts the application on localhost accessible via a web browser.
        """
        # Execute the command to run the application
        os.system("streamlit run whowrotethis//whowrotethis_app.py")
