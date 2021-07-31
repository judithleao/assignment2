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

MY NEXT TO DOS:

Store the category_names list in the db as well and import in run.py
Store the language_detect table in db and import when compiling chart as it currently takes too long and stops website from loading
Change html layout to 4 (or three, potentially remove template chart) rows with your charts and add headers
Improve actual model (run overnight?)
