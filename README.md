Here's a GitHub-friendly README.md file for your face recognition system:

```markdown
# Face Recognition System

## Overview

This project is a face recognition system built with Python, utilizing the Tkinter library for the graphical user interface, OpenCV for real-time video capture, and the `face_recognition` library for facial recognition. It also uses SQLite to manage user profiles and their face embeddings.

## Features

- Register New Users: Capture and register new users by providing their roll number, name, and food preference. Facial images are used for recognition.
- Remove Users: Delete users from the database by roll number.
- Clear Database: Remove all user records from the database.
- Real-time Face Recognition: Recognize faces in real-time from webcam feed and display associated user details.
- Train Model: Update face embeddings for existing users to improve recognition accuracy.

## Requirements

- Python 3.x
- `tkinter` (usually included with Python)
- `opencv-python`
- `numpy`
- `Pillow`
- `face_recognition`
- `sqlite3` (usually included with Python)

To install the required packages, run:
```
```bash
pip install opencv-python numpy Pillow face_recognition
```

## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/face-recognition-system.git
   cd face-recognition-system
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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact [your-email@example.com](mailto:your-email@example.com).

```

Replace `https://github.com/your-username/face-recognition-system.git` with the actual URL of your GitHub repository and `[your-email@example.com](mailto:your-email@example.com)` with your contact email. This README provides a comprehensive overview, setup instructions, usage details, and contribution guidelines for your project.
