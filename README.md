# Osu Cheat Detection based on Tensorflow

## Requirements
* Tensorflow
* Pandas
* sklearn
* osrparse
* keras
* requests
* argparse

## Installation
This project uses virtualenv. When you are in the Root Library, use `.\Scripts\activate`
or any version of activate, that you want to use.

To install all requirements as needed, use 
> pip install -r Requirements.txt

## Development
`resetModel.py` generates default models without training data.

Any pullrequests are appreciated, just fork, branch, commit + pull request. I will do my best to keep up to date.

## Usage
When in your Root directory, use `python ./Source/main.py -h` to get a full view of the help window.

**NOTE** If not providing `-legit`, it will assume all replays provided are cheated.

Everything should be self explaining.