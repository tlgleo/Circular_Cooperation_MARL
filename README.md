# Circular Social Dilemma with Reinforcement Learning

## Requirements
- Python: requirements.txt
- graphbasedTFT and circular_collect (see Installation)
- Other: ffmpeg (for video rendering)

## Installation

To create a virtual environment:

* Download files on your machine and go to the main directory
  ```
  git clone https://github.com/submission-conf/neurips_tmp.git
  cd neurips_tmp
  ```

* Create a virtual environment and activate it
  ```
  python3 -m venv RL_circular_cooperation
  source RL_circular_cooperation/bin/activate
  ```

* Load the python librairies needed from the requirements file
  ```
  python -m pip install --upgrade pip
  python -m pip install --upgrade setuptools
  python -m pip install -r requirements.txt
  ```


* Install our package graphbasedTFT from previous work (under review), the graph-based Tit-for-Tat. 
  ```
  git clone https://github.com/submission-conf/graphbasedTFT.git
  cd graphbasedTFT
  pip3 install -e .
  cd ..
  ```

* Install our Gym circular game, the package circular_collect:
  ```
  git clone https://github.com/submission-conf/circular_games.git
  cd circular_games/circular_collect
  pip3 install -e .
  cd ../..
  ```


## Lauching a game
Some simulations of games with our algorithm are available in the script `main_demo.py`.
Rendering videos are possible if ffmpeg is installed.

### Customized simulation TODO
Select `choice_example = 0` and modify below:

### Presets simulations TODO
Select the index of example in `choice_example = `
Here are the proposed simulations:
1. 

## Evaluation TODO
The script `main_evaluation.py` computes some metrics in some games

1. 
2.
