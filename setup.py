import os

os.path.normpath(os.path.abspath(__file__) + os.sep + os.pardir)

if __name__ == "__main__":
    cwd = os.getcwd()
    print(cwd)