# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Notes
```
You are right, there are many different names that can be used to refer to the 'target variable' in machine learning.

Here are some of the most common terms you may come across:

    Target variable
    Dependent variable
    Response variable
    Outcome variable
    Label
    Class

Each of these terms refers to the variable we are trying to predict or estimate using the input data. The choice of terminology may depend on the specific field of study, the nature of the problem being addressed, or personal preference.

Regardless of the term used, the target variable is a fundamental component of a supervised learning problem, where the goal is to train a machine learning model to make accurate predictions of the target variable based on the input data.
```