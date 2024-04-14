"""
    DS 5010 - Final Project
    Error Logger - a function to log errors to ai_human_errors.err
    Created by Xin Wang
"""

import datetime

FILE_NAME = "ai_human_errors.err"


def get_time():
    """FUNCTION: get_time - RETURN: current time"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_error(error_message, function_name, code_file,
              file_name=FILE_NAME):
    """
    FUNCTION: log_error
    This is a function to log errors

    PARAMETERS:
    error_message - str
        - the error message when encountering the error

    function_name - str
        - the function from which the error came out

    code_file - str
        - the name of the file in which the function is included

    file_name - str
        - the name of the file to which the error message will be written
    """

    try:
        with open(file_name, encoding="utf-8", mode="a") as infile:
            infile.write(f"{get_time()} - ERROR: {error_message} " +
                         f"- ERROR from {function_name} " +
                         f"in file {code_file}\n")

    except Exception as error:
        print(f"Encounter error: {error}, " +
              f"fail to log the error: {error_message} " +
              f"from {function_name} in file {code_file}")


