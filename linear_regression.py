import os
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


def getData():
    # lấy dữ liệu từ file csv
    dataFile = None
    if os.path.exists('home_data.csv'):
        print("-- home_data.csv found locally")
        dataFile = pd.read_csv('home_data.csv', skipfooter=1, engine='python')
    print(dataFile)
    return dataFile


def linearRegressionModel(X_train, Y_train, X_test, Y_test):
    linear = linear_model.LinearRegression()
    # Training model
    linear.fit(X_train, Y_train)
    # Evaluating the model
    score_trained = linear.score(X_test, Y_test)  # điểm đánh giá mô hình
    # Input
    num_bed = int(input('Enter the number of bedrooms: '))
    year_built = int(input('Enter the year the house was built: '))
    living_area = int(input('Enter the living area (in square feet): '))
    num_room = int(input('Enter the number of rooms: '))
    num_bath = int(input('Enter the number of bathrooms: '))

    input_model = [[num_bed, year_built, living_area,
                    num_room, num_bath]]  # dạng ma trận
    # lấy ra giá nhà dự đoán dựa trên model đã huấn luyện
    predicted_price = linear.predict(np.array(input_model))

    return predicted_price


if __name__ == "__main__":
    data = getData()
    if data is not None:
        # Selection few attributes
        attributes = list(
            [
                'num_bed',
                'year_built',
                'num_room',
                'num_bath',
                'living_area',
            ]
        )
        # Vector price of house
        Y = data['askprice']
        # print np.array(Y)
        # Vector attributes of house
        X = data[attributes]
        # Split data to training test and testing test
        X_train, X_test, Y_train, Y_test = train_test_split(
            np.array(X), np.array(Y), test_size=0.2)
        # Linear Regression Model
        predicted_price = linearRegressionModel(
            X_train, Y_train, X_test, Y_test)
        price = round(predicted_price[0])
        formatted_price = '$'+'{:,.0f}'.format(
            price).replace(',', '.')
        print('price = ', formatted_price)
