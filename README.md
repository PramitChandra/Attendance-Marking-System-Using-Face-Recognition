# Attendance Marking System Using Face Recognition

This project implements a simple attendance marking system using face recognition. It captures video input from a camera, detects faces in the frames, recognizes the faces, and marks attendance in a SQLite database. The project uses the LBPH (Local Binary Patterns Histograms) face recognition algorithm provided by OpenCV.

## Overview

- The project trains the LBPH face recognizer using images of individuals in a specified directory.
- It uses the trained model to recognize faces in real-time video frames.
- Recognized faces are matched to names in a label dictionary, and attendance is recorded in a SQLite database.
- Minimum confidence level for face recognition can be set to filter out low-confidence predictions.

## Prerequisites

- Python 3.x
- OpenCV
- SQLite3

## Usage

1. Prepare a directory with subdirectories, each containing images of a person you want to recognize.
2. Update the `path` variable in the code to point to the directory with the face images.
3. Run the Python code.
4. The system will recognize faces in the video stream and mark attendance in the SQLite database.
5. Press 'q' to exit the system.

## Note

- Ensure that you have the necessary Python packages installed (OpenCV and SQLite3).
- Adjust the `path` variable in the code to match the directory containing face images.
- You may need to adjust the `confidence_threshold` to control the sensitivity of face recognition.

## Author

Pramit Chandra

## License

This project is licensed under the [MIT License](LICENSE).
