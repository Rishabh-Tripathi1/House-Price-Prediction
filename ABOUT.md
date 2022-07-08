# House Price Prediction
**A website where user can get an estimate price of a house at certain places in Mumbai by entering certain requirements of the user. <br />The website is connected to a Machine Learning model that is trained on a dataset containing prices of different property all across Mumbai.**

## About the website:
The website is a single page HTML coded frontend which is connected on the backend with the help of flask.

Components on the site:
1. Location of the property (Drop-Down)
2. Area in Sq/Ft (Input)
3. No. of Bedrooms (Input)
4. Toggle Buttons
     - Gymnasium
     - Car Parking
     - Indoor Games
     - Jogging Track

## About the model:
The model has been trained on the dataset "[Housing Prices in Mumbai](https://www.kaggle.com/datasets/sameep98/housing-prices-in-mumbai)". The training algorithm in use is [RandomForestRegresson](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).
The dataset was broken into **70/30** ratio with random state of 0.

The model has an **R2_Score: 0.82**

## Requirements: 
Following libraries are required:
1. Sci-kit Learn (Sklearn)
2. Pandas
3. Flask
4. Numpy

## ScreenShots:
### Home:
![alt text](https://github.com/Rishabh-Tripathi1/House-Price-Prediction/blob/main/images/Home.jpg?raw=true)

### Prediction:
![alt text](https://github.com/Rishabh-Tripathi1/House-Price-Prediction/blob/main/images/Prediction.jpg?raw=true)
