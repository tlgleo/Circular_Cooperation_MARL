# Circular Social Dilemma with Reinforcement Learning

## Requirements
- Python: requirements.txt
- Other: ffmpeg

## Installation of packages
Our algorithm needs another package from previous work (under review), the graph-based Tit-for-Tat. 
To install it:

```
git clone https://github.com/submission-conf/graphbasedTFT.git
cd graphbasedTFT
pip3 install -e .
cd ..
```

Moreover, our algorithm is run on games we design. In particular the Circular Collect.
To install it:

```
git clone https://github.com/submission-conf/circular_games.git
cd circular_collect/circular_collect
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
