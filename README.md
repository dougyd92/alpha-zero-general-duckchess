# Implementing Duck Chess with Alpha Zero General

Building upon the training framework found at https://github.com/suragnair/alpha-zero-general,
this repo adds an implementation [Duck Chess](https://duckchess.com/) as a learnable game.

## Which contributions are new?
- duckchess/DuckChessNN.py and duckchess/DuckChessNetWrapper.py. This adds the neural network that will be used to learn both the policy and value for the game. It uses a 12-block residual tower, with policy and values heads as described in (Silver et al. 2017).
- duckchess/DuckChessLogic.py. This adds the rules of Duck Chess, state and aciton encoding, state transitions, etc.
- duckchess/DuckChessGame.py and duckchess/DuckChessPlayers.py. This implements the API in Game.py in order to fit into the training framework.
- compare_to_random.py, head_to_head.py, human_vs_ai.py. Alternatives to pit.py to facilitate qualitative and quantitative analysis of different model iterations.

## What modifications were made to existing code?
- Coach.py. Modified the training algorithm to continously train a single model, rather than comparing models each iteration and taking the best. This matches the changes made to the training algorithm between AlphaGo-Zero and AlphaZero.
- MCTS.py. Fixed a memory leak where the old portion of the tree wasn't being discard, leading to linear growth in memory usage and crashes for games with a large number of turns.
- main.py / Coach.py. Added an arg 'starting_iteration' to avoid overwriting previous examples and models when restarting from a checkpoint. 