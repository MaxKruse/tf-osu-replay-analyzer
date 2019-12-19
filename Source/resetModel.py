import tensorflow
from tensorflow import keras

import os


print("You are about to reset the models used for antiCheat detection.")

validate = input("To reset relax.model, type 'delete_relax.model'")
if validate is 'delete_relax.model':
	if os.path.exists("relax.model"):
		os.remove('relax.model')
	model = tensorflow.keras.Sequential()
	model.add(keras.layers.Flatten(500)) # 500 Inputs are, worst case, samples with 10 seconds. Usually its samples for ~6-7 seconds though
	model.add(keras.layers.Dense(40, activation='relu')) # single complexity, not very good
	model.add(keras.layers.Dense(80, activation='relu')) # double complexity, enough for basic understanding
	model.add(keras.layers.Dense(160, activation='relu')) # tripple complexity, enough for complex understanding
	model.add(keras.layers.Dense(320, activation='relu')) # quad complexity, enough for everything
	model.add(keras.layers.Dense(2), activation='softmax') # Output, softmax to go from 0.0 to 1.0

	model.save("relax.model")
	model.summary()
	print("Reset relax.model")
else:
	print("Skipping relax.model")

validate = input("To reset aim.model, type 'delete_aim.model'")
if validate is 'delete_aim.model':
	if os.path.exists("aim.model"):
		os.remove('aim.model')
		model = tensorflow.keras.Sequential()

	model.add(keras.layers.Flatten(500)) # 500 Inputs are, worst case, samples with 10 seconds. Usually its samples for ~6-7 seconds though
	model.add(keras.layers.Dense(40, activation='relu')) # single complexity, not very good
	model.add(keras.layers.Dense(80, activation='relu')) # double complexity, enough for basic understanding
	model.add(keras.layers.Dense(160, activation='relu')) # tripple complexity, enough for complex understanding
	model.add(keras.layers.Dense(320, activation='relu')) # quad complexity, enough for everything
	model.add(keras.layers.Dense(2), activation='softmax') # Output, softmax to go from 0.0 to 1.0

	model.save("aim.model")
	model.summary()
	print("Reset aim.model")
else:
	print("Skipping aim.model")

print("Finished resetting Models")