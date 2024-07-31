

# Face Recognition System

## Overview

This project is a face recognition system developed in Python. It integrates a graphical user interface (GUI) using Tkinter, real-time video capture with OpenCV, and facial recognition through the `face_recognition` library. User data, including facial embeddings, is stored and managed using an SQLite database.

## Features

- **Register New Users:** Add new users by capturing their facial image and entering their roll number, name, and food preference.
- **Remove Users:** Delete existing users from the database using their roll number.
- **Clear Database:** Erase all user records from the database.
- **Real-time Face Recognition:** Perform real-time face recognition using webcam feed and display associated user details.
- **Train Model:** Update face embeddings for all users to improve recognition accuracy.

## Requirements

- Python 3.x
- `tkinter` (usually included with Python)
- `opencv-python`
- `numpy`
- `Pillow`
- `face_recognition`
- `sqlite3` (usually included with Python)

Install the necessary packages with:

```bash
pip install opencv-python numpy Pillow face_recognition
```
## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/Face-recognition-system.git
   cd Face-recognition-system
   ```

2. **Setup the Database:**

   The application will automatically create an SQLite database file named `user_profiles.db` in the project directory.

3. **Run the Application:**

   Start the application by running:

   ```bash
   python face_recognition_app.py
   ```

   This will open the GUI window for user interaction.

## Usage

### Register New User

1. Enter the roll number, name, and food preference in the respective fields.
2. Click "Capture & Register" to capture a face image and store the details in the database.

### Remove User

1. Enter the roll number of the user to be removed.
2. Click "Remove User" to delete the user from the database.

### Clear Database

1. Click "Clear Database" to remove all user records from the database.

### Real-time Face Recognition

1. Click "Start Real-time Recognition" to begin video capture and real-time face recognition.

### Train Model

1. Click "Train Model" to update the face embeddings for all registered users with the latest captured images.

## Notes

- Ensure your webcam is properly connected and accessible for image capture.
- You may need to adjust the face recognition threshold in the `compare_face_embedding` function based on your dataset.
- Handle exceptions and errors to ensure a smooth user experience.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

## Contact

For questions or feedback, please contact [your-email@example.com](mailto:parayu2905@gmail.com).

```
