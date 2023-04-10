## Table of Contents
- [Description](#description)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Installing](#installing)
- [Executing Program](#executing-program)
- [Screenshots](#screenshots)

## Description
The objective is to construct a Natural Language Processing (NLP) model capable of categorizing messages from real-life disaster events. The dataset consists of pre-labeled tweets and messages. The project is divided into three main sections. 

    Data processing:
        Develop an ETL pipeline to extract data from the source
        Clean the data
        Save the cleaned data in a SQLite database

    Machine learning pipeline:
        Build a machine learning pipeline
        Train the model to classify text messages into various categories

    Web application:
        Run a web application to display the results of the model in real-time.

## Getting Started
1. run Piplines/ETL Pipeline Preparation.ipynb and will make ETL_Preparation.db file
2. run Piplines/ML Pipeline Preparation.ipynb and will make mo2.pkl file
3. in models/train_classifier.py is doing the samething as ML Pipeline Preparation.ipynb, it creates mo2.pkl file

## Dependencies

requirements.txt

- numpy
- pandas
- sqlalchemy<2.0
- NLTK
- scikit-learn
- plotly
- Flask

## Installing

```
python -m venv myenv

# for windows
myenv\Scripts\activate.bat

# for linux and macOS
source myenv/bin/activate

pip install -r requirements.txt
```

## Executing Program
Making the database file
```
cd data
python process_data.py disaster_messages.csv disaster_categories.csv ETL_Preparation.db
```

Making the 'pickle file'
```
cd models && python train_classifier.py ../Piplines/ETL_Preparation.db mo3.pkl
cd app && python run.py 
```

Run the 'Flask application'
```
cd app
python run.py
```
go to http://127.0.0.1:3001/


## Screenshots
![alt text](screenshots/1.png)
![alt text](screenshots/1.png)
![alt text](screenshots/1.png)
![alt text](screenshots/etl_database.png)







