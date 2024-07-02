'''module to predict the price of a car based on the mileage'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json


class PredictionService:
    '''class to predict the price of a car based on the mileage'''
    def __init__(self) -> None:
        '''initializes the class with the theta0 and theta1 values'''
        self.df = None
        self.theta0, self.theta1 = 0, 0
        self.users_prediction = 0
        self.predicted_price = 0

    def start(self) -> None:
        self.df = pd.read_csv('data/data.csv')
        self.theta0, self.theta1 = self.getTheta()
        self.users_prediction = self.getUsersPrediction()
        self.predicted_price = self.calculatePredictedPrice()
        self.plotData()

    def calculatePredictedPrice(self) -> float:
        '''returns the predicted price of the car'''
        predicted_price = self.theta0 + self.theta1 * self.users_prediction
        if predicted_price < 0:
            raise ValueError('The predicted price is too big')
        print(f'The predicted price of the car is: {predicted_price}$')
        return predicted_price

    def getTheta(self) -> tuple:
        '''returns the theta0 and theta1 values from the model'''
        data = json.load(open('model.json'))
        theta0 = data['theta0']
        theta1 = data['theta1']
        return theta0, theta1

    def getUsersPrediction(self) -> float:
        '''returns the predicted price of the car based on the mileage'''
        number = float(input('Enter the mileage of the car: '))
        if number < 0:
            raise ValueError('Mileage cannot be negative')
        return number

    def plotData(self) -> None:
        '''plots the data points in the data.csv file'''

        line = np.array([0, - self.theta0 / self.theta1])
        plt.plot(line, self.theta0 + self.theta1 * line, 'r')
        plt.plot(self.df['km'], self.df['price'], 'o')
        plt.plot(self.users_prediction, self.predicted_price, 'go')

        plt.legend(['Model', 'Data points', 'User prediction'])
        plt.title('Price of cars based on mileage')

        plt.xlabel('Mileage (in thousands of km)')
        plt.gca().get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{int(x / 1000)}')
        )

        plt.ylabel('Price')
        plt.gca().get_yaxis().set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{int(y / 1000)}k')
        )

        plt.show()


def main() -> None:
    '''entrypoint of the script
    prompts the user to enter the mileage of the car
    predicts the price of the car based on theta0 and theta1 from the model
    '''
    try:
        prediction_service = PredictionService()
        prediction_service.start()
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
