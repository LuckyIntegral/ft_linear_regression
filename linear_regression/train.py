'''module to train the model based on the dataset'''
import pandas as pd
import numpy as np
import json
from tqdm import tqdm


class LinearRegression:
	'''class to train the model based on the dataset'''
	def __init__(self, datafile: str) -> None:
		'''initializes the class'''
		self.df_km, self.df_price, self.M = self.getData(datafile)
		self.intercept = 0
		self.coef = 0
		self.learning_rate = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

	def normalize(self) -> None:
		'''normalizes the data'''
		self.max_km = np.max(self.df_km)
		self.min_km = np.min(self.df_km)
		self.max_price = np.max(self.df_price)
		self.min_price = np.min(self.df_price)

		self.df_km = (self.df_km - self.min_km) / (self.max_km - self.min_km)
		self.df_price = (self.df_price - self.min_price) / (self.max_price - self.min_price)

	def denormalizeTheta(self, theta1: float, theta0: float) -> tuple:
		'''denormalizes the theta0 and theta1 values'''
		theta1 = theta1 * (self.max_price - self.min_price) / (self.max_km - self.min_km)
		theta0 = theta0 * (self.max_price - self.min_price) + self.min_price - theta1 * self.min_km
		return theta1, theta0

	def getData(self, datafile: str) -> tuple:
		'''returns the mileage and price data from the dataset'''
		df = pd.read_csv(datafile)
		return df['km'], df['price'], len(df)

	def train(self) -> None:
		'''trains the model based on the dataset'''

		self.normalize()
		print('Training the model...')

		for iteration in tqdm(range(1000 * len(self.learning_rate))):
			lr = self.learning_rate[iteration // 1000]
			predictions = self.estimatePrice(self.df_km)
			self.intercept -= lr / self.M * np.sum(predictions - self.df_price)
			self.coef -= lr / self.M * np.sum((predictions - self.df_price) * self.df_km)

		print('Training completed')
		self.coef, self.intercept = self.denormalizeTheta(self.coef, self.intercept)

	def estimatePrice(self, mileage: float) -> float:
		'''returns the estimated price of the car based on the mileage'''
		return self.intercept + self.coef * mileage

	def saveModel(self) -> None:
		'''saves the theta0 and theta1 values in a file'''
		data = {'theta0': self.intercept, 'theta1': self.coef}
		with open('model.json', 'w') as file:
			json.dump(data, file)


def main() -> None:
	'''entrypoint of the script
	trains the model based on the dataset
	stores the theta0 and theta1 in file model.json
	'''
	try:
		lr = LinearRegression('data/data.csv')
		lr.train()
		lr.saveModel()
	except Exception as e:
		print(e)
		sys.exit(1)

if __name__ == '__main__':
	main()
