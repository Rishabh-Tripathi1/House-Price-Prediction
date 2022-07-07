import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
data = pd.read_csv('train.csv')
model = pickle.load(open('model.pkl','rb'))
le = LabelEncoder()
x = le.fit_transform(data['Location'])

@app.route('/') 
def index():

    locations = sorted(data['Location'].unique())
    return render_template('home.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    area = request.form.get('area')
    bhk = request.form.get('bhk')
    ne = request.form.get('toggle')
    gy = request.form.get('gym')
    ind = request.form.get('ind')
    ca = request.form.get('car')
    jog = request.form.get('jog')
    n=0
    for i in range(414):
         if data['Location'][i] == location:
            n = i
            break
    if gy == 'on':
        gym = 1
    else:
        gym = 0
    if jog == 'on':
        jogg = 1
    else:
        jogg = 0
    if ca == 'on':
        car = 1
    else:
        car = 0
    if ind == 'on':
        indd = 1
    else:
        indd = 0
    if ne == 'on':
        new = 1
    else:
        new = 0
    print(area,x[n],bhk,new,gym,car,indd,jogg)
    input = pd.DataFrame([[area,x[n],bhk,new,gym,car,indd,jogg]], columns=['Area','Location','No. of Bedrooms','New/Resale','Gymnasium','Car Parking','Indoor Games','Jogging Track'])
    pred = model.predict(input)[0]*1e6
    return str(np.round(pred,2))


if __name__ == "__main__":
    app.run(debug = True, port=5000)