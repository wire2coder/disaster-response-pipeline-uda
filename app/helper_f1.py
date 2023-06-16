# function to check if file exist
def file_exist1(file_path):
    import os

    if os.path.exists(file_path):
        return True
    else:
        return False

