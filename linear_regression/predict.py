'''module to predict the price of a car based on the mileage'''
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def main() -> None:
	'''entrypoint of the script
	prompts the user to enter the mileage of the car
	then predicts the price of the car based on theta0 and theta1 from the model
	'''
	try:
		df = pd.read_csv('data/data.csv')

		user_input = float(input('Enter the mileage of the car: '))

		mpl.use('Qt5Agg')
		plt.plot(df['km'], df['price'], 'o')
		plt.show()

	except Exception as e:
		print(e)

if __name__ == '__main__':
	main()
