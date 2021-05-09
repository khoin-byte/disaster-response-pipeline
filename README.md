# Disaster Response Pipeline Project
Project goal was to classify disaster tweet messages - a multi-label classification problem.

# Repo/Content Structure
* Root directory contains jupyter notebooks for both Extract Transform Load (ETL) and ML Pipeline.
* data: both categories and messages csv files, consumed by process_data.py for ETL pipeline.
* models: containes main train_classifier.py to reload sql db from ETL pipeline and model building. Saving model as .pkl.
* app: Python Flask is use to serve the web app via run.py

# Run Instructions
* To run ETL pipeline that cleans data and stores in database ** python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv** 
* To run ML pipeline that trains classifier and saves **python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl**
* To run the app, Run the following command in the app's directory to run your web app. **python run.py**. Then go to: **http://0.0.0.0:3001/**

