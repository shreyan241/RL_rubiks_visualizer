# Reinforcement Learning Rubiks Cube Visualizer and Solver

## Description
This project features a 3D Rubik's Cube visualizer and an advanced reinforcement learning-based solver using the Monte Carlo Tree Search (MCTS) algorithm. The visualizer was developed in Python with matplotlib, leveraging quaternion mathematics to rotate the cube in 3D space. The Rubik’s Cube visualizer was built using the [MagicCube repo](https://github.com/davidwhogg/MagicCube/tree/master) written by David W. Hogg and Jacob Vanderplas. It also implements a modified form of the ADI (Autodidactic Iteration) algorithm for solving the Rubik's Cube, based on the research paper: [Solving the Rubik’s Cube Without Human Knowledge](https://arxiv.org/abs/1805.07470).

## Demo
To see the project in action, check out the demo video available in the Demo folder of this repository.
![Rubik's Cube Demo](Demo/output.gif)

## Installation & Setup
This setup is intended for Windows OS. You need Python 3.11 or higher and Poetry version 1.3.8. If Poetry isn’t installed, run.bat will install it for you.

1. Prerequisites
  - Option 1: If you already have Python 3.11 or higher, skip to running 'run.bat'.
  - Option 2: Create a Conda environment for Python 3.11:
'''
conda create -n rubiks_cube_solver python=3.11
conda activate rubiks_cube_solver
'''

2. Running the Visualizer
  - Run the run.bat file - 'run.bat'
  - This will:
      - Install Poetry if not present.
      - Set up dependencies.
      - Start the server and launch the visualizer.
