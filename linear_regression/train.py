'''module to train the model based on the dataset'''
import pandas as pd
import numpy as np
import json
import sys


def main() -> None:
	'''entrypoint of the script
	trains the model based on the dataset
	stores the theta0 and theta1 in file model.json
	'''
	try:
		df = pd.read_csv('data/data.csv')
		theta0, theta1 = 0, 0
		json.dump({'theta0': theta0, 'theta1': theta1}, open('model.json', 'w'))
	except Exception as e:
		print(e)

if __name__ == '__main__':
	main()
