import os
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score


def getData(fileData):
    # lấy dữ liệu từ file csv
    url = fileData + '.csv'
    dataFile = None
    if os.path.exists(url):
        print("-- home_data.csv found locally")
        dataFile = pd.read_csv(url, delimiter=';', engine='python')

        # tính toán ma trận tương quan giữa các biến
        # corr_matrix = dataFile.corr()
        # print(corr_matrix)
        # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    return dataFile


# Phân tích hệ số phụ thuộc của từng yếu tố vào giá nhà
def coefficient(linear, X):
    # Vẽ biểu đồ
    fig, ax = plt.subplots()
    rects = ax.bar(X.columns, linear.coef_)
    print(linear.coef_)
    plt.xlabel('Các yếu tố')
    plt.ylabel('Hệ số phụ thuộc')
    plt.title('Sơ đồ biểu thị hệ số phụ thuộc của các yếu tố vào giá nhà')

    # Thêm giá trị vào biểu đồ
    for i, rect in enumerate(rects):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                round(linear.coef_[i], 2),
                ha='center', va='bottom')
    plt.show()


def correlation_matrix(dataFile, X):
    columns_name = dataFile.columns.tolist()
    corr_matrix = dataFile.corr()
    print('Corr_matrix:')
    print(corr_matrix)
    elements = {}
    for i in range(0, len(columns_name)):
        elements[columns_name[i]] = corr_matrix['price'][i]

    sorted_elements = dict(sorted(
        elements.items(), key=lambda x: x[1], reverse=True))
    for element in sorted_elements:
        print(element+":\t"+str(sorted_elements[element]))
    # Vẽ biểu đồ
    key_list = list(elements.keys())
    value_list = list(elements.values())
    fig, ax = plt.subplots()
    rects = ax.bar(key_list, value_list, color='blue')
    # Hiển thị tên các key trên trục x
    ax.set_xticks(key_list)

    # Đặt góc xoay cho các nhãn trên trục x
    plt.xticks(rotation=45)
    plt.xlabel('Các yếu tố')
    plt.ylabel('Hệ số phụ thuộc')
    plt.title('Sơ đồ biểu thị hệ số phụ thuộc của các yếu tố vào giá nhà')
    # Thêm giá trị vào biểu đồ
    for i, rect in enumerate(rects):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                round(corr_matrix['price'][i], 2),
                ha='center', va='bottom')
    plt.show()


def linearRegressionModel(X_train, Y_train, X_test, Y_test):
    linear = linear_model.LinearRegression()
    # Training model
    linear.fit(X_train, Y_train)
    return linear
# evaluate the dependencies
# def evaluate_dependencies(linear,element):


def R_squared(linear, X_test, Y_test):
    y_pred = linear.predict(X_test)
    score_trained = linear.score(X_test, Y_test)  # điểm đánh giá mô hình
# Tính toán R-squared trên tập kiểm tra
    r2 = r2_score(Y_test, y_pred)

    print('R-squared:', r2)
    print('Score: '+str(score_trained))


def predict_price():
    data = getData('dataset')
    if data is not None:
        # Selection few attributes
        attributes = list(
            [
                'area',
                'bathroom',
                'bedroom',
                'longtitude',
                'latitude',
                'cleaningfee'
            ]
        )

        # Vector price of house
        Y = data['price']
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
        return formatted_price


if __name__ == "__main__":
    data = getData('dataset')
    data2 = getData('dataset1')
    if data is not None:
        # Selection few attributes
        attributes = list(
            [
                'area',
                'bathroom',
                'bedroom',
                'latitude',
                'longtitude',
            ]
        )
        # Vector price of house
        Y = data['price']
        # print np.array(Y)
        # Vector attributes of house
        X = data[attributes]
        # Split data to training test and testing test
        X_train, X_test, Y_train, Y_test = train_test_split(
            np.array(X), np.array(Y), test_size=0.2)
        linear = linearRegressionModel(
            X_train, Y_train, X_test, Y_test)

        print('Phân tích các yếu tố ảnh hưởng tới giá nhà: \n1. Dùng hệ số phụ thuộc \n2. Sử dụng ma trận tương quan \n3. Tính điểm của mô hình \n4. Thoát')
        # đánh giá hệ số phụ thuộc
        while (True):
            choice = input('Lựa chọn của bạn: ')
            if choice == '1':
                coefficient(linear, X)
            elif (choice == '2'):
                # đánh giá ma trận tương quan
                correlation_matrix(data, X)
            elif (choice == '3'):
                R_squared(linear, X_test, Y_test)
            else:
                break
