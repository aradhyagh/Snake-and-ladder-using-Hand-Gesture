# Snake-and-ladder-using-Hand-Gesture

## Features
1. Classic Snake game rendered using OpenCV with a configurable NxN grid
2. Real-time gameplay using keyboard controls (W/A/S/D)
3. Optional hand-gesture-based control using MediaPipe (index finger or thumb gestures)
4. Simultaneous display of game window and live webcam feed
5. Randomly generated fruits and obstacle walls with user-defined counts
6. Snake head highlighted separately from the body
7. Score tracking displayed on the game window
8. Pause and resume functionality with keyboard control
9. Game over screen with final score and restart/quit options
10. Customizable sound effects for fruit collection and collisions

## Tech Stack

Python 3

OpenCV

MediaPipe (Tasks API)

NumPy

winsound (for audio playback on Windows)

---

## Controls:

W / A / S / D: Move snake and resume game from pause

P: Pause or resume the game

Q: Quit the game

Hand gestures: Control snake direction based on the configured gesture mode

---

## Configuration

Key configuration options are defined at the top of main.py:

GRID_SIZE: Size of the game grid (NxN)

CELL_SIZE: Pixel size of each grid cell

NUM_FRUITS: Number of fruits on the board

NUM_WALLS: Number of obstacle walls

CONTROL_MODE: Keyboard, hand gesture, or both

MODEL_PATH: Path to the MediaPipe hand landmark model

Sound file paths for fruit and collision events

---

## Future Improvements

The gesture are to be more regulated and controlable
