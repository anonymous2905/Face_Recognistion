import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import sqlite3
import face_recognition

# Initialize SQLite database
DB_FILE = 'user_profiles.db'

def setup_database():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                roll_number TEXT UNIQUE,
                name TEXT,
                embedding BLOB,
                food_preference TEXT
            )
        ''')
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")
    finally:
        if conn:
            conn.close()

def add_user_to_db(roll_number, name, embedding, food_preference):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO users (roll_number, name, embedding, food_preference)
            VALUES (?, ?, ?, ?)
        ''', (roll_number, name, sqlite3.Binary(embedding.tobytes()), food_preference))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        print(f"Roll number '{roll_number}' already exists in the database.")
        return False
    except Exception as e:
        print(f"Failed to add user: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def recognize_face(face_image):
    try:
        face_locations = face_recognition.face_locations(face_image)
        face_encodings = face_recognition.face_encodings(face_image, face_locations)
        
        if not face_encodings:
            return None
        
        face_descriptor = face_encodings[0]
        recognized_roll_number = compare_face_embedding(face_descriptor)
        return recognized_roll_number
    except Exception as e:
        print(f"Failed to recognize face: {str(e)}")
        return None

def compare_face_embedding(query_embedding):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('SELECT roll_number, embedding FROM users')
        rows = cursor.fetchall()

        if len(rows) == 0:
            print("No users in the database.")
            return None

        best_match_roll_number = None
        min_distance = float('inf')

        for row in rows:
            roll_number, stored_embedding_blob = row
            stored_embedding = np.frombuffer(stored_embedding_blob, dtype=np.float64)

            distance = np.linalg.norm(query_embedding - stored_embedding)

            if distance < min_distance:
                min_distance = distance
                best_match_roll_number = roll_number

        # Adjust this threshold as needed
        if min_distance < 0.6:  # Example threshold, adjust based on your dataset
            return best_match_roll_number
        else:
            return None

    except Exception as e:
        print(f"Failed to compare embeddings: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("Face Recognition System")
        master.geometry("1000x800")

        self.create_widgets()

    def create_widgets(self):
        # Registration Frame
        register_frame = ttk.LabelFrame(self.master, text="Register New User")
        register_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(register_frame, text="Roll Number:").grid(row=0, column=0, padx=5, pady=5)
        self.entry_roll_number = ttk.Entry(register_frame)
        self.entry_roll_number.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(register_frame, text="Name:").grid(row=1, column=0, padx=5, pady=5)
        self.entry_name = ttk.Entry(register_frame)
        self.entry_name.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(register_frame, text="Food Preference:").grid(row=2, column=0, padx=5, pady=5)
        self.entry_food_preference = ttk.Entry(register_frame)
        self.entry_food_preference.grid(row=2, column=1, padx=5, pady=5)

        ttk.Button(register_frame, text="Capture & Register", command=self.register_user).grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        self.label_register_image = ttk.Label(register_frame)
        self.label_register_image.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

        # Remove User Frame
        remove_frame = ttk.LabelFrame(self.master, text="Remove User")
        remove_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(remove_frame, text="Roll Number:").grid(row=0, column=0, padx=5, pady=5)
        self.entry_remove_roll = ttk.Entry(remove_frame)
        self.entry_remove_roll.grid(row=0, column=1, padx=5, pady=5)

        ttk.Button(remove_frame, text="Remove User", command=self.remove_user).grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        # Database Actions Frame
        db_frame = ttk.LabelFrame(self.master, text="Database Actions")
        db_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Button(db_frame, text="Clear Database", command=self.clear_database).grid(row=0, column=0, padx=5, pady=5)

        # Add Train Model Button
        ttk.Button(db_frame, text="Train Model", command=self.train_model).grid(row=1, column=0, padx=5, pady=5)

        # Recognition Frame
        recog_frame = ttk.LabelFrame(self.master, text="Face Recognition")
        recog_frame.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")

        ttk.Button(recog_frame, text="Start Real-time Recognition", command=self.start_recognition).grid(row=0, column=0, padx=5, pady=5)

        self.label_video = ttk.Label(recog_frame)
        self.label_video.grid(row=1, column=0, padx=5, pady=5)

        self.label_result = ttk.Label(recog_frame, text="Recognition Result")
        self.label_result.grid(row=2, column=0, padx=5, pady=5)

        self.label_recognized_image = ttk.Label(recog_frame)
        self.label_recognized_image.grid(row=3, column=0, padx=5, pady=5)

    def register_user(self):
        roll_number = self.entry_roll_number.get()
        name = self.entry_name.get()
        food_preference = self.entry_food_preference.get()

        if not roll_number or not name or not food_preference:
            messagebox.showerror("Error", "Roll Number, Name, and Food Preference are required")
            return

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Error", "Failed to capture image")
            return

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        if not face_encodings:
            messagebox.showerror("Error", "No face detected")
            return

        face_descriptor = face_encodings[0]

        face_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_image_pil = Image.fromarray(face_image_rgb)
        photo = ImageTk.PhotoImage(face_image_pil)
        self.label_register_image.config(image=photo)
        self.label_register_image.image = photo

        if add_user_to_db(roll_number, name, face_descriptor, food_preference):
            messagebox.showinfo("Success", "User added to database")
            self.entry_roll_number.delete(0, tk.END)
            self.entry_name.delete(0, tk.END)
            self.entry_food_preference.delete(0, tk.END)
        else:
            messagebox.showerror("Error", "Failed to add user to database")

    def remove_user(self):
        roll_number = self.entry_remove_roll.get()
        if not roll_number:
            messagebox.showerror("Error", "Please enter Roll Number")
            return

        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM users WHERE roll_number = ?', (roll_number,))
            conn.commit()

            if cursor.rowcount == 0:
                messagebox.showinfo("Not Found", "No user with this Roll Number found")
            else:
                messagebox.showinfo("Success", "User removed from database")

        except sqlite3.Error as e:
            messagebox.showerror("Error", f"Failed to remove user: {e}")
        finally:
            if conn:
                conn.close()

    def clear_database(self):
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM users')
            conn.commit()
            messagebox.showinfo("Success", "Database cleared")
        except sqlite3.Error as e:
            messagebox.showerror("Error", f"Failed to clear database: {e}")
        finally:
            if conn:
                conn.close()

    def start_recognition(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            recognized_roll_number = recognize_face(frame)
            if recognized_roll_number:
                conn = sqlite3.connect(DB_FILE)
                cursor = conn.cursor()
                cursor.execute('SELECT name, food_preference FROM users WHERE roll_number = ?', (recognized_roll_number,))
                row = cursor.fetchone()
                conn.close()

                if row:
                    name, food_preference = row
                    self.label_result.config(text=f"Name: {name}, Food Preference: {food_preference}")
                else:
                    self.label_result.config(text="User not found")
            else:
                self.label_result.config(text="Face not recognized")

            face_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_image_pil = Image.fromarray(face_image_rgb)
            photo = ImageTk.PhotoImage(face_image_pil)
            self.label_video.config(image=photo)
            self.label_video.image = photo

            self.master.update()
            self.master.after(10)

        cap.release()

    def train_model(self):
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute('SELECT roll_number, name FROM users')
            rows = cursor.fetchall()

            for row in rows:
                roll_number, name = row

                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                cap.release()

                if not ret:
                    messagebox.showerror("Error", "Failed to capture image")
                    continue

                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                
                if not face_encodings:
                    messagebox.showerror("Error", "No face detected")
                    continue

                face_descriptor = face_encodings[0]

                cursor.execute('''
                    UPDATE users SET embedding = ? WHERE roll_number = ?
                ''', (sqlite3.Binary(face_descriptor.tobytes()), roll_number))
                conn.commit()

            messagebox.showinfo("Success", "Model trained on current dataset")

        except sqlite3.Error as e:
            messagebox.showerror("Error", f"Failed to train model: {e}")
        finally:
            if conn:
                conn.close()

if __name__ == "__main__":
    setup_database()
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
