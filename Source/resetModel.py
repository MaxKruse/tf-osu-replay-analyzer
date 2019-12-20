import tensorflow
import keras

import os


print("You are about to reset the models used for antiCheat detection.")

validate = input("To reset relax.model, type 'delete_relax.model': ").strip()
if validate == 'delete_relax.model':
	if os.path.exists("relax.model.h5"):
		os.remove('relax.model.h5')
	model = tensorflow.keras.Sequential()

	model.add(tensorflow.keras.layers.Input((500,), name="positions")) # 500 Inputs are, worst case, samples with 10 seconds. Usually its samples for ~6-7 seconds though.
	model.add(tensorflow.keras.layers.Dense(40, activation='relu', name="low_complex")) # single complexity, not very good
	model.add(tensorflow.keras.layers.Dense(80, activation='relu', name="mid_complex")) # double complexity, enough for basic understanding
	model.add(tensorflow.keras.layers.Dense(160, activation='relu', name="high_complex")) # tripple complexity, enough for complex understanding
	model.add(tensorflow.keras.layers.Dense(320, activation='relu', name="all_complex")) # quad complexity, enough for everything
	model.add(tensorflow.keras.layers.Dense(units=2, activation='softmax', name="result")) # Output, softmax to go from 0.0 to 1.0

	model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

	model.save("relax.model.h5")
	model.summary()
	print("Reset relax.model")
else:
	print("Skipping relax.model")

validate = input("To reset aim.model, type 'delete_aim.model': ").strip()
if validate == 'delete_aim.model':
	if os.path.exists("aim.model.h5"):
		os.remove('aim.model.h5')
	model = tensorflow.keras.Sequential()

	model.add(tensorflow.keras.layers.Input((500,2), name="positions")) # 500 Inputs are, worst case, samples with 10 seconds. Usually its samples for ~6-7 seconds though. x2 for x + y
	model.add(tensorflow.keras.layers.Dense(40, activation='relu', name="low_complex")) # single complexity, not very good
	model.add(tensorflow.keras.layers.Dense(80, activation='relu', name="mid_complex")) # double complexity, enough for basic understanding
	model.add(tensorflow.keras.layers.Dense(160, activation='relu', name="high_complex")) # tripple complexity, enough for complex understanding
	model.add(tensorflow.keras.layers.Dense(320, activation='relu', name="all_complex")) # quad complexity, enough for everything
	model.add(tensorflow.keras.layers.Dense(units=2, activation='softmax', name="result")) # Output, softmax to go from 0.0 to 1.0

	model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

	model.save("aim.model.h5")
	model.summary()
	print("Reset aim.model")
else:
	print("Skipping aim.model")

print("Finished resetting Models")