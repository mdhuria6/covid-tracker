from flask import Flask, render_template , request, jsonify
import joblib
import pandas as pd
import numpy as np
import datetime

app = Flask(__name__)

total_case = open("total_case_model.pkl","rb")
total_case_model = joblib.load(total_case)

total_death = open("total_deaths_model.pkl","rb")
total_deaths_model = joblib.load(total_death)

@app.route("/")

def predict():
    print("This is the homepage")

    d = datetime.date(year=2020,month=9,day=10)
    print(d);
    d = d.toordinal()
    print(d);
    d = np.array(d).reshape(-1,1)
    print(d);
    total_cases = total_case_model.predict(d)

    total_cases = total_cases.round();
    total_cases = str(total_cases).lstrip('[').rstrip('.]')

    total_deaths = total_deaths_model.predict(d)

    total_deaths = total_deaths.round()
    total_deaths = str(total_deaths).lstrip('[]').rstrip('.]')

    # return render_template('index.html',display = [total_cases,total_deaths])
    datafinal = [{'total_cases' : total_cases, 'total_deaths' : total_deaths}]
    return jsonify(datafinal);

if __name__ == "__main__":
    app.run()
