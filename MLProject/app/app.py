from flask import Flask, render_template, request
import numpy as np
from joblib import load 
import pickle

rfc = pickle.load(open("loan.pkl","rb"))

#initializing app
app = Flask(__name__) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == "POST":
        data1 = request.form["Married"]
        if data1 == "Yes":
            data1 = 1
        else:
            data1 = 0

        data2 = request.form["Gender"]
        if data2 == "Male":
            data2 = 1
        else:
            data2 = 0

        data3 = request.form["Dependents"]
        if data3 == "0":
            data3 = 0
        elif data3 == "1":
            data3 = 1
        elif data3 == "2":
            data3 = 2
        else:
            data3 = 3  

        data4 = request.form["Education"]
        if data4 == "Yes":
            data4 = 1
        else:
            data4 = 0   

        data5 = request.form["Self-Employed"]
        if data5 == "Yes":
            data5 = 1
        else:
            data5 = 0  

        data6 = request.form["Applicant-Income"]

        data7 = request.form["Co-Applicant-Income"]

        data8 = request.form["Loan-Amount"]

        data9 = request.form["Loan-Amount-Term"]

        data10 = request.form["Credit-History"]
        if data10 == "All Debts Paid":
            data10 = 1
        else:
            data10 = 0

        data11 = request.form["Property-Area"]
        if data11 == "Urban":
            data11 = 2
        elif data11 == "Semiurban":
            data11 = 1
        else:
            data11 = 0 
        
        input = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11]])

        prediction = rfc.predict(input)

        # if prediction == 1: #Eligible for a loan
        #     return render_template("index.html", text = "Congratualations, you are eligible for a loan with Banking Beaver!")
        # else:
        #     return render_template("index.html", text = "Unfortunately, you are not eligible for a loan with Banking Beaver.")

        return render_template("result.html", data = prediction)
    else:
        return render_template("index.html")


app.run(debug=True)