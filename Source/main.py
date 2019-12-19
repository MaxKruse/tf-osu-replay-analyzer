from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys
import os
import requests, tempfile

def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    parser = argparse.ArgumentParser(description="Machine Learning using TensorFlow to find out if replays are cheated in any way. Really, ANY way.")

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("-replay",
                    help="Path to the replay file. Can both be filesystem or http/https")
    group.add_argument("-list", 
                    help="Give a path to a file with locations of replay files. Works with mixed Filesystem and URLs")

    parser.add_argument("-verbose", required=False, action="store_true",
                    help="Literally print everything")
    parser.add_argument("-legit", required=False, action="store_true", default=0,
                    help="Assume the replay is legit, for training only")
    parser.add_argument("-train", required=False, action="store_true", default=False,
                    help="Marks the provided replays 'to be used to train'. Evaluates if omitted")
    return parser

arg_parser = create_arg_parser()
parsed_args = arg_parser.parse_args(sys.argv[1:])

if not os.path.exists("aim.model") or not os.path.exists("relax.model"):
    print("Models do not exist. Please run resetModel.py to re-create them.")
    exit(1)


# Set Constant Path
pathToFile = "{0}/replay.osr".format(tempfile.gettempdir())
pathToCsv  = "{0}/replay.csv".format(tempfile.gettempdir())

# Loading all Dependencies to start doing some processing at all
print("Loading osrParse and Pandas")

import osrparse
from osrparse.enums import GameMode

import pandas as pd
import numpy as np

def cache_replay(path):
    # Deleting old files
    if parsed_args.verbose:
            print("Removing previous temporary files")

    if os.path.exists(pathToFile):
        os.remove(pathToFile)

    if os.path.exists(pathToCsv):
        os.remove(pathToCsv)

    # Getting files
    if not os.path.exists(path):
        replay = requests.get(path)

        if replay.status_code is not 200:
            print("File doesnt exist or cant be downloaded. Exiting...")
            return

        with open(pathToFile, "wb") as f:
            print("Saving replay file")
            f.write(replay.content)
    else:
        with open(path, "rb") as g:
            with open(pathToFile, "wb") as f:
                print("Saving replay file")
                f.write(g.read())

    print("Replay file used: \"{0}\".\nGot from \"{1}\"".format(pathToFile, path))

    print("Loading Replay file")
    replay = osrparse.parse_replay_file(pathToFile)

    # Raw Data to CSV
    print("Getting Raw Data from Replay")
    x = []
    y = []
    deltaTime = []
    realTime = []
    totalTime = 0
    keys = []
    target = []

    t = 1

    if parsed_args.legit:
        t = 0

    #   Standard = 0
    #   Taiko = 1
    #   CatchTheBeat = 2
    #   Osumania = 3

    if replay.game_mode is not GameMode.Standard:
        print("Only supporting std replays (0). Found: {}".format(replay.game_mode))
        return

    for playEvent in replay.play_data:
        x.append(playEvent.x)
        y.append(playEvent.y)
        deltaTime.append(playEvent.time_since_previous_action)
        keys.append(playEvent.keys_pressed)
        totalTime += playEvent.time_since_previous_action
        realTime.append(totalTime)
        target.append(t)

    dataframe = pd.DataFrame({
        "x": x, 
        "y": y,
        "deltaTime": deltaTime,
        "realTime": realTime,
        "keys": keys,
        "cheated": target
        })

    print("Saving Raw Gameplay data to csv")
    dataframe.to_csv(pathToCsv)

    if parsed_args.verbose:
        print(dataframe)

if parsed_args.list:
    print("From text file")
    with open(parsed_args.list) as file:
        for cnt, line in enumerate(file):
            cache_replay(line.strip())

elif parsed_args.replay:
    print("From single replay file")

def train_map(path):
    print("Not Yet Implemented")

    
# Begin Tensorflow
print("Loading tensorflow and sklearn")
from sklearn.model_selection import train_test_split
import tensorflow
import keras

# functions

# A utility method to create a data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=False, batch_size=128):
    dataframe = dataframe.copy()
    labels = dataframe.pop("cheated")
    if parsed_args.verbose:
        print("Labels:", labels)
    ds = tensorflow.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
    return ds

aim_model = keras.models.load_model('aim.model')
print("Loaded Aim Model: ")
aim_model.summary()

relax_model = keras.models.load_model('relax.model')
print("Loaded Relax Model: ")
relax_model.summary()