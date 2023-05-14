from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import mysql.connector
import json
#demo softmax
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import io
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


app = Flask(__name__)
CORS(app)

#demo softmax load model
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# Load model
model = tf.keras.models.load_model('mnist_model.h5')


# kết nối database để đọc thông tin địa chỉ sưu tập
mydb = mysql.connector.connect(
    host="localhost", user="root", password="29012002", database="homedata"
)

# Tìm giá trị gần với giá dự đoán nhất
def find_nearest_value(prices, target):
    nearest_value = None
    min_distance = None

    for price in prices:
        distance = abs(price - target)
        if min_distance is None or distance < min_distance:
            nearest_value = price
            min_distance = distance

    return nearest_value


#Tìm ra 3 ngôi nhà có giá gần với giá dự đoán nhất
def findThreeHouseNearest(prices, target_price):
    nearest_prices = []
    # Tìm ra 3 giá ngôi nhà có giá phù hợp nhất
    for i in range(3):
        nearest_price = find_nearest_value(prices, target_price)
        nearest_prices.append(nearest_price)
        prices.remove(nearest_price)
    return nearest_prices

#So sánh giá các ngôi nhà với 3 giá trị tìm được
def HasSamePrice(prices, price_predict):
    for p in prices:
        if p == price_predict:
            return True
    return False

#Gửi về thông tin các ngôi nhà gần với yêu cầu người dùng nhất
def findSuitableHouse(data, price_predict):
    home = {}
    result = []
    prices = data["price"].tolist()
    nearest_prices = findThreeHouseNearest(prices, price_predict)
    price, num_bath, num_room, area = 0, 0, 0, 0
    latitude, longtitude = "", ""
    for i in range(len(data["price"])):
        if HasSamePrice(nearest_prices, data["price"][i]):
            nearest_prices.remove(data["price"][i])
            home = {}
            price, num_bath, num_room, area = (
                data["price"][i],
                data["bathroom"][i],
                data["bedroom"][i],
                data["area"][i],
            )
            latitude, longtitude = str(data["latitude"][i]), str(data["longtitude"][i])
            # thêm dữ liệu để chuẩn bị gửi đi
            formatted_price = "{:,.0f}".format(price_predict).replace(",", ".") + " VNĐ"
            # Tạo json file để gửi về client
            home["price_predict"] = formatted_price
            home["price"] = "{:,.0f}".format(price).replace(",", ".") + " VNĐ"
            home["num_bath"] = str(num_bath)
            home["num_room"] = str(num_room)
            home["area"] = str(area)
            home["pos"] = getPosition(latitude, longtitude)
            result.append(home)
    return jsonify(result)


# đọc dữ liệu
def getData():
    # lấy dữ liệu từ file csv
    dataFile = None
    if os.path.exists("dataset.csv"):
        print("-- home_data.csv found locally")
        dataFile = pd.read_csv("dataset.csv", delimiter=";", engine="python")
    return dataFile

#Đưa ra giá nhà dự đoán dựa trên các yếu tố đầu vào
def linearRegressionModel(X_train, Y_train, X_test, Y_test, input_model):
    linear = linear_model.LinearRegression()
    # Training model
    linear.fit(X_train, Y_train)
    y_pred = linear.predict(X_test)
    # Tính toán R-squared trên tập kiểm tra
    r2 = r2_score(Y_test, y_pred)
    print("score:", r2)
    predicted_price = linear.predict(np.array(input_model))
    return predicted_price


data = getData()

# train model

#Train model
def RegressionModel():
    if data is not None:
        # Selection few attributes
        attributes = list(
            [
                "area",
                "bathroom",
                "bedroom",
                "longtitude",
                "latitude",
            ]
        )

        # Vector price of house
        Y = data["price"]
        # print np.array(Y)
        # Vector attributes of house
        X = data[attributes]
        # Split data to training test and testing test
        X_train, X_test, Y_train, Y_test = train_test_split(
            np.array(X), np.array(Y), test_size=0.2, random_state=0
        )
        linear = linear_model.LinearRegression()
        # Training model
        linear.fit(X_train, Y_train)
        y_pred = linear.predict(X_test)
        # Tính toán R-squared trên tập kiểm tra
        r2 = r2_score(Y_test, y_pred)
        print("score:", r2)
        return linear


# dự đoán giá nhà
def predict_price(input_model, longtitude, latitude):
    predicted_price = linear.predict(np.array(input_model))
    price = round(predicted_price[0])

    formatted_price = "{:,.0f}".format(price).replace(",", ".") + " VNĐ"
    return price


linear = RegressionModel()#khởi tạo model linear regression

# trả về danh sách địa điểm các quận huyện ở hà đông
def getPosition(latitude, longtitude):
    mycursor = mydb.cursor()
    mycursor.execute(
        "SELECT name FROM address where latitude = %s and longtitude = %s",
        (latitude, longtitude),
    )
    data = mycursor.fetchall()
    result = "".join(data[0])
    return result

#Trả về giao diện trang chủ
@app.route("/")
def home():
    mycursor = mydb.cursor()
    mycursor.execute("SELECT name, longtitude, latitude FROM address")
    data = mycursor.fetchall()
    return render_template("index.html", data=data)


#Xử lý các dữ liệu từ client và gửi đi
@app.route("/api", methods=["POST"])
def sendRespone():
    if request.method == "POST":
        # Lấy dữ liệu từ client
        response = request.get_json()
        square = int(response["area"])
        bedroom = int(response["Bedroom"])
        bathroom = int(response["Bathroom"])
        longtitude = int(response["longtitude"])
        latitude = int(response["latitude"])
        input_model = [[square, bedroom, bathroom, longtitude, latitude]]
        price = predict_price(input_model, longtitude, latitude)
        # Xử lý dữ liệu và trả về kết quả cho client
        return findSuitableHouse(data, price)
 
#trang web demo softmax
@app.route('/demo')
def demo():
    # Xử lý logic của trang demo ở đây
    return render_template('demo.html')

#trả về dự đoán số
@app.route("/predict", methods=["POST"])
def sendPredict():
    file = request.files['image']
    img_bytes = file.read()

    # Preprocess image
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_number = np.argmax(prediction[0])

    return jsonify({'prediction': str(predicted_number)})

if __name__ == "__main__":
    app.run()
