import tensorflow
import keras

import os

def calc_neurons(current, final_output):
	return int(current * (2 / 3) + (final_output * 2) + 32)

print("You are about to reset the models used for antiCheat detection.")

validate = input("To reset relax.model, type 'delete_relax.model': ").strip()
if validate == 'delete_relax.model':
	if os.path.exists("relax.model.h5"):
		os.remove('relax.model.h5')
	model = tensorflow.keras.Sequential()

	neurons = 500

	model.add(tensorflow.keras.layers.Input((neurons,), name="positions")) # 500 Inputs are, worst case, samples with 10 seconds. Usually its samples for ~6-7 seconds though.
	neurons = calc_neurons(neurons, 2)
	model.add(tensorflow.keras.layers.Dense(neurons, activation='relu', name="low_complex")) # single complexity, not very good
	neurons = calc_neurons(neurons, 2)
	model.add(tensorflow.keras.layers.Dense(neurons, activation='relu', name="mid_complex")) # double complexity, enough for basic understanding
	neurons = calc_neurons(neurons, 2)
	model.add(tensorflow.keras.layers.Dense(neurons, activation='relu', name="high_complex")) # tripple complexity, enough for complex understanding
	neurons = calc_neurons(neurons, 2)
	model.add(tensorflow.keras.layers.Dense(neurons, activation='relu', name="all_complex")) # quad complexity, enough for everything
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

	neurons = 500 * 2

	model.add(tensorflow.keras.layers.Input((500,2), name="positions")) # 500 Inputs are, worst case, samples with 10 seconds. Usually its samples for ~6-7 seconds though. x2 for x + y
	neurons = calc_neurons(neurons, 2)
	model.add(tensorflow.keras.layers.Dense(neurons, activation='relu', name="low_complex")) # single complexity, not very good
	neurons = calc_neurons(neurons, 2)
	model.add(tensorflow.keras.layers.Dense(neurons, activation='relu', name="mid_complex")) # double complexity, enough for basic understanding
	neurons = calc_neurons(neurons, 2)
	model.add(tensorflow.keras.layers.Dense(neurons, activation='relu', name="high_complex")) # tripple complexity, enough for complex understanding
	neurons = calc_neurons(neurons, 2)
	model.add(tensorflow.keras.layers.Dense(neurons, activation='relu', name="all_complex")) # quad complexity, enough for everything
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