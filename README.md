# Udacity Data Scientist Nandodegree Program
## Disaster Response Pipeline Project
Project goal was to classify disaster tweet messages (courtesy of Figure Eight) - a multi-label classification problem.

# Repo/Content Structure
* Root directory contains jupyter notebooks for both Extract Transform Load (ETL) and ML Pipeline.
* data: both categories and messages csv files, consumed by process_data.py for ETL pipeline.
* models: containes main train_classifier.py to reload sql db from ETL pipeline and model building. Saving model as .pkl.
* app: Python Flask is use to serve the web app via run.py

# How to Use
Dependency
Main code base should run Python 3.* with the following library (pip install if needed):
* Numpy, Pandas, Sqlite3, SQLalchemy, Scikit-Learn, Pickle, NLTK for ETL and ML pipelines.
* Flask, Plotly for Flask web app.

# Run Instructions
* To run ETL pipeline that cleans data and stores in database ** python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv** 
* To run ML pipeline that trains classifier and saves **python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl**
* To run the app, Run the following command in the app's directory to run your web app. **python run.py**. Then go to: **http://0.0.0.0:3001/**

# Web App
Main Entry to Web App with search bar for user to type in query for inference.
![main page](https://github.com/khoin-byte/disaster-response-pipeline/blob/main/demo/entry.png)

Result of running inference on trained model.
![search page](https://github.com/khoin-byte/disaster-response-pipeline/blob/main/demo/search_results.png)

# Future Improvements
From the bar plots below of a few lables, we can clearly see an imbalance of the minority class. This is a problem and we can see that for the F1-score metric, some of the heavily imbalanced labels have very low F1-score. This inevitably affected the overall macro F1-score. Some techniques to improve results:
* Some models, e.g. RandomForestClassifier and Logistic Regression have a **class_weight** parameter that we can set as 'balanced' to put more weight on the minority class.
* Imbalance Learn, i.e. imblearn has random over sampling and Synthetic Minority Oversampling Technique to augment our dataset and increase the minority class.
![imbalance](https://github.com/khoin-byte/disaster-response-pipeline/blob/main/demo/imbalance_lables.png)
