# dueling-ddqn-atari-rl
Implementation of the Dueling-Double-Deep-Q-Learning algorithm with [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) from scratch. Solution was iteratively extended from my [DQN](https://arxiv.org/abs/1312.5602) implementation, to include:
- [Double Q Learning](https://arxiv.org/pdf/1509.06461) - a second value network with delayed weight updates to perform the bootstrap estimate of the expected cumulative reward (i.e. error term) to reduce value overestimation
- [Dueling Networks](https://arxiv.org/abs/1511.06581) - replacing the direct projection of state-action Q-values with separate projections for state-value and action advantage respectively

The DDQN training and testing was performed in both OpenAI gym and Atari environments, however Dueling DDQN was only trained for OpenAI gym environments due to limited compute. 

Original goal is to expand upon this algorithm until it reaches the [Rainbow DQN](https://arxiv.org/pdf/1710.02298) architecture from DeepMind.
