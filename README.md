# Disaster Response Pipeline Project


## Motivation:
This model classifies messages/news sent out during disasters into topic categories that are relevant for different NGOs. This allows faster filtering of relevant messages and thus improves response time.
On the website, statistics of the original data (test and training combined) can be viewed in charts. You can enter a new message into the bar and classify the message. This uses the model trained in train_classifier.py.


## Libraries needed:
* flask
* langdetect
* nltk
* numpy
* pandas (version 0.24.0 or higher)
* pickle
* plotly
* plotly.express
* re
* sklearn
* sqlalchemy
* sys


## Files needed:
Input datafiles: 
- disaster_categories.csv
- disaster_messages.csv
Python scripts:
- process_data.py (Pre-processes data, creates database)
- train_classifier.py (Builds and trains model, saves pickle file with model parameters)
- run.py (Compiles web-app)
- length_estimator.py (Module called in train_classifier.py model pipeline)
HTML files:
- master.html (Basic website structure)
- go.html (Blocks that replace elements of master.html on click of button)


## Files that get created when running code:
- DisasterResponse.db (contains 2 tables: Messages_Table: data model is built on; Categories_Table: All labels for dependent variable)
- classifier.pkl (contains model parameters)


## Input requirements:
If new input data was to be used it needs to follow the formatting of the existing input data. Specifically the messages.csv needs to contain at least a *message* column, an *original* column. The categories.csv needs to contain a *category* column that contains all categories with binary values separated by ";" in each cell and no missing data. The script will remove records with nan in the *message* column and will recode *category* columns with values >1 as 0. Both files need to contain indices that match for records that belong together and the same number of records.


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. The website will run on port 3001. To run it in the Udacity workspace, please run 'env | grep WORK' and find the website under http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN


## Licensing and citations:
The data was kindly provided via the Udacity course project by FigureEight (now Appen; website: https://appen.com/).
Any citations for code snippets are provided directly in the code in the approriate places.
Some code was kindly provided by Udacity as part of the course project.

### Citation for libraries:
* flask: Grinberg, M. (2018). Flask web development: developing web applications with python. " O&#x27;Reilly Media, Inc."
* langdetect: PyPi package by Nakatani Shuyo, slide show here: https://www.slideshare.net/shuyo/language-detection-library-for-java
* nltk: Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
* numpy: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 0.1038/s41586-020-2649-2. 
* pandas (version 0.24.0 or higher): Jeff Reback, jbrockmendel, Wes McKinney, Joris Van den Bossche, Tom Augspurger, Phillip Cloud, Simon Hawkins, gfyoung, Sinhrks, Matthew Roeschke, Adam Klein, Terji Petersen, Jeff Tratner, Chang She, William Ayd, Patrick Hoefler, Shahar Naveh, Marc Garcia, Jeremy Schendel, … Kaiqi Dong. (2021). pandas-dev/pandas: Pandas 1.3.1 (v1.3.1). Zenodo. https://doi.org/10.5281/zenodo.5136416
* pickle: Van Rossum, G. (2020). The Python Library Reference, release 3.8.2. Python Software Foundation.
* plotly/plotly.express: Plotly Technologies Inc. Title: Collaborative data science Publisher: Plotly Technologies Inc. Place of publication: Montréal, QC Date of publication: 2015 URL: https://plot.ly
* re: Van Rossum, G. (2020). The Python Library Reference, release 3.8.2. Python Software Foundation.
* sklearn: Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
* sqlalchemy: Michael Bayer. SQLAlchemy. In Amy Brown and Greg Wilson, editors, The Architecture of Open Source Applications Volume II: Structure, Scale, and a Few More Fearless Hacks 2012 http://aosabook.org
* sys: Van Rossum, G. (2020). The Python Library Reference, release 3.8.2. Python Software Foundation.



