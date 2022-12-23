# Implementing Duck Chess with Alpha Zero General

Building upon the training framework found at https://github.com/suragnair/alpha-zero-general,
this repo adds an implementation [Duck Chess](https://duckchess.com/) as a learnable game.

## Which contributions are new?
- duckchess/DuckChessNN.py and duckchess/DuckChessNetWrapper.py. This adds the neural network that will be used to learn both the policy and value for the game. It uses a 12-block residual tower, with policy and values heads as described in (Silver et al. 2017).
- duckchess/DuckChessLogic.py. This adds the rules of Duck Chess, state and aciton encoding, state transitions, etc.
- duckchess/DuckChessGame.py and duckchess/DuckChessPlayers.py. This implements the API in Game.py in order to fit into the training framework.