# Chess_robot_IEEE
This repository contains the source code and resources for a chess-playing robot. The project integrates artificial intelligence, gameplay prediction, evaluation, and robotic control to deliver a dynamic and engaging chess-playing experience. The AI is supported by Stockfish, a powerful open-source chess engine.

This project is designed to bring together the realms of AI and robotics through chess. It enables a robotic arm to play chess against a human or AI opponent by integrating AI-based predictions, gameplay evaluation, and real-time robotic control.

Features

Stockfish Integration: Supports AI gameplay with Stockfish for move calculations.

Game Prediction and Evaluation: Evaluates and predicts optimal moves for both AI and human players.

Robotic Control: Controls the physical movement of the robot to play chess on a physical board.

Customizable Game Settings: Flexible for human vs. robot.

Project Structure

README.md: Documentation file for the repository.

chess.py: Contains the logic for chess rules, board representation, and move validation.

chessgame.py: Manages the main gameplay loop and integrates AI and robotic functions.

game_data.py: Stores and processes data related to gameplay states and history.

game_prediction&evaluation.py: Handles prediction algorithms and evaluates game positions using Stockfish.

robot_control.py: Interfaces with the robotic hardware to execute physical chess moves.

stockfishX64.exe: Stockfish chess engine executable.
