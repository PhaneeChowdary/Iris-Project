# Import all dependencies
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from flask import Flask, redirect, render_template, request, url_for, Response

# WSGI Application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

# Form inputs
@app.route('/submit', methods=["GET", "POST"])
def submit():
    if request.method == "POST":
        Lengths = []
        seplen = float(request.form['SepalLength'])
        sepwid = float(request.form['SepalWidth'])
        petlen = float(request.form['PetalLength'])
        petwid = float(request.form['PetalWidth'])
        lengths = np.array([seplen, sepwid, petlen, petwid]).reshape(1,-1)
    
        dataframe = pd.read_csv('Iris.csv')
        df = dataframe.copy()
        df.drop('Id', axis = 1, inplace = True)
        input_cols = df.select_dtypes(include=['int64', 'float64']).columns
        target_col = 'Species'
        species_code = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        df['Species'].replace(species_code, inplace=True)
        inputs_df = df[input_cols]
        targets = df[target_col]

        train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs_df,
                                                                                    targets,
                                                                                    test_size=0.25,
                                                                                    random_state=42)
        
        #RandomForestClassifier
        rfc_model = RandomForestClassifier(random_state=42)
        rfc_model.fit(train_inputs, train_targets)
        test_preds = rfc_model.predict(test_inputs)

        classes = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
        rfc_predict = rfc_model.predict(lengths)
        res = classes[rfc_predict[0]]
        accuracy = str(accuracy_score(test_targets, test_preds)*100) + "%"

    return render_template("prediction.html", result = res, accuracy = accuracy)

if __name__ == "__main__":
    app.run(debug = True)