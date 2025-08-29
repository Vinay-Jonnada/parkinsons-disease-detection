import tkinter as tk
from tkinter import messagebox, scrolledtext
import os
import json
import subprocess
import sys
import numpy as np
import joblib



# === Application Configuration ===
# This script is the GUI for the Parkinson's Detection App.
# It depends on:
# 1. 'svm_model.joblib' and 'scaler.joblib' (from train_model.py)
# 2. 'model_performance.json' (also from train_model.py) containing
#    the model's accuracy, precision, recall, and f1_score.

# === Constants ===
PROJECT_DIR = 'D:/parkinsons_project'
if not os.path.exists(PROJECT_DIR):
    os.makedirs(PROJECT_DIR)
    
USER_DATA_FILE = os.path.join(PROJECT_DIR, 'users.json')
DATASET_FILE_PATH = os.path.join(PROJECT_DIR, 'parkinsons.csv')
MODEL_FILE = os.path.join(PROJECT_DIR, 'svm_model.joblib')
SCALER_FILE = os.path.join(PROJECT_DIR, 'scaler.joblib')
PERFORMANCE_FILE = os.path.join(PROJECT_DIR, 'model_performance.json') # New file for metrics

# The exact list of features the model was trained on, in the correct order.
FEATURES_FOR_MODEL = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", 
    "MDVP:Shimmer", "Shimmer:DDA", "HNR", "RPDE", "DFA", 
    "spread1", "spread2", "D2", "PPE"
]

# === Global Variables ===
current_user = None
prediction_entries = {}
model_performance = {} # To store loaded metrics

# --- Load the trained model, scaler, and performance metrics at startup ---
try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    model_loaded = True
    print("INFO: Model and scaler loaded successfully.")

    # Load performance metrics
    try:
        with open(PERFORMANCE_FILE, 'r') as f:
            model_performance = json.load(f)
        print("INFO: Model performance metrics loaded successfully.")
    except (FileNotFoundError, json.JSONDecodeError):
        model_performance = {} # Set default empty values
        print(f"WARNING: Performance file not found or invalid. Metrics will show as 0.00.")

except FileNotFoundError:
    model, scaler, model_loaded, model_performance = None, None, False, {}
    print("WARNING: Model or scaler file not found. Prediction feature will be disabled.")


# === Ensure users.json exists ===
if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump({}, f)

# === Main App Window ===
root = tk.Tk()
root.title("Parkinson's Detection App")
root.geometry("800x750") # Give it a bit more space
root.config(bg="#f0f2f5") # A neutral background like the image's surrounding

# === Frames ===
home_frame = tk.Frame(root)
menu_frame = tk.Frame(root, bg="#e6f2ff")
login_frame = tk.Frame(root, bg="#fdf6ec")
signup_frame = tk.Frame(root, bg="#fdf6ec")
abstract_frame = tk.Frame(root, bg="#f9f9f9")
algorithm_frame = tk.Frame(root, bg="#f9f9f9")
dataset_info_frame = tk.Frame(root, bg="#f9f9f9")
help_frame = tk.Frame(root, bg="#f9f9f9")
welcome_frame = tk.Frame(root, bg="#e0f7fa")
predict_frame = tk.Frame(root, bg="#fffde7")
profile_frame = tk.Frame(root, bg="#f3e5f5")
result_frame = tk.Frame(root, bg="#f0f2f5")


def hide_all_frames():
    for frame in [home_frame, menu_frame, login_frame, signup_frame, abstract_frame, algorithm_frame, dataset_info_frame, help_frame, welcome_frame, predict_frame, profile_frame, result_frame]:
        frame.pack_forget()

def open_path(path):
    if not os.path.exists(path):
        messagebox.showerror("Path Not Found", f"The specified path does not exist:\n{path}")
        return
    try:
        if sys.platform == "win32": os.startfile(path)
        elif sys.platform == "darwin": subprocess.run(["open", path])
        else: subprocess.run(["xdg-open", path])
    except Exception as e:
        messagebox.showerror("Error", f"Could not open the path:\n{e}")

def show_home():
    hide_all_frames()
    home_frame.pack(fill='both', expand=True)
    for widget in home_frame.winfo_children(): widget.destroy()
    try:
        from PIL import Image, ImageTk
        img_path = os.path.join(PROJECT_DIR, 'home.png')
        original_image = Image.open(img_path)
        bg_image = ImageTk.PhotoImage(original_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.LANCZOS))
        background_label = tk.Label(home_frame, image=bg_image); background_label.image = bg_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
    except Exception as e:
        home_frame.config(bg="#f0f0f0")
        tk.Label(home_frame, text="Parkinson's Disease Detection", font=("Georgia", 28, "bold"), bg="#f0f0f0").pack(pady=(200, 20))
        tk.Label(home_frame, text=f"ERROR: Could not load 'home.png'.\n{e}", fg="red", bg="#f0f0f0").pack()
    tk.Button(home_frame, text="Proceed →", command=show_menu, bg="#28a745", fg="white", font=("Arial", 16, "bold"), relief="raised", borderwidth=3, padx=20, pady=8).place(relx=0.5, rely=0.85, anchor='center')
    tk.Button(home_frame, text="Exit", command=root.destroy, bg="#dc3545", fg="white", font=("Arial", 14, "bold"), relief="raised", borderwidth=3, padx=15, pady=5).place(relx=0.5, rely=0.93, anchor='center')

def show_menu():
    hide_all_frames(); menu_frame.pack(fill='both', expand=True);
    for w in menu_frame.winfo_children(): w.destroy();
    main_content_frame = tk.Frame(menu_frame, bg="#e6f2ff"); main_content_frame.pack(fill='both', expand=True, padx=20, pady=10)
    left_frame = tk.Frame(main_content_frame, bg="#e6f2ff"); left_frame.pack(side=tk.LEFT, fill='y', padx=(80, 20))
    tk.Label(left_frame, text="MAIN MENU", font=("Arial", 20, "bold"), bg="#e6f2ff").pack(pady=(20, 15))
    button_block = tk.Frame(left_frame, bd=2, relief="groove", bg="white", padx=30, pady=25); button_block.pack(anchor='n', expand=True, pady=(30, 0))
    menu_buttons = [("Login / Sign Up", show_login_signup), ("Abstract", show_abstract), ("Algorithm & Example", show_algorithm), ("Dataset Information", show_dataset_info), ("Help", show_help)]
    for text, cmd in menu_buttons: tk.Button(button_block, text=text, command=cmd, width=25, font=("Arial", 12), bg="#007bff", fg="white", pady=7).pack(pady=10)
    right_frame = tk.Frame(main_content_frame, bg="#e6f2ff"); right_frame.pack(side=tk.RIGHT, fill='both', expand=True, padx=(20, 10))
    try:
        from PIL import Image, ImageTk
        img_path = os.path.join(PROJECT_DIR, 'menu_icon.png')
        resized_image = Image.open(img_path).resize((900, 700), Image.LANCZOS)
        menu_image = ImageTk.PhotoImage(resized_image)
        image_label = tk.Label(right_frame, image=menu_image, bg="#e6f2ff"); image_label.image = menu_image
        image_label.pack(anchor='center', expand=True)
    except Exception as e:
        tk.Label(right_frame, text="[Image Not Found]", font=("Arial", 12), bg="#e6f2ff", fg="gray").pack(anchor='center', expand=True)
        print(f"ERROR loading menu_icon.png: {e}")
    tk.Button(menu_frame, text="⬅ Back to Home", command=show_home, bg="#6c757d", fg="white", font=("Arial", 12, "bold"), padx=15, pady=6).pack(pady=(0, 20), side=tk.BOTTOM)

def show_login_signup():
    hide_all_frames(); login_frame.pack(fill='both', expand=True)
    for widget in login_frame.winfo_children(): widget.destroy()
    tk.Label(login_frame, text="User Login", font=("Arial", 18, "bold"), bg="#fdf6ec").pack(pady=20)
    tk.Label(login_frame, text="Username", font=("Arial", 11), bg="#fdf6ec").pack(pady=(10,0)); username_entry = tk.Entry(login_frame, font=("Arial", 11), width=30); username_entry.pack()
    tk.Label(login_frame, text="Password", font=("Arial", 11), bg="#fdf6ec").pack(pady=(10,0)); password_entry = tk.Entry(login_frame, show="*", font=("Arial", 11), width=30); password_entry.pack()
    def login_action():
        global current_user; username, password = username_entry.get(), password_entry.get()
        if not username or not password: messagebox.showerror("Error", "Please enter both username and password"); return
        with open(USER_DATA_FILE, 'r') as f: users = json.load(f)
        if username in users and users[username] == password:
            current_user = username; messagebox.showinfo("Login Successful", f"Welcome, {current_user}!"); show_welcome_page()
        else: messagebox.showerror("Login Failed", "Invalid username or password")
    def show_signup_page():
        hide_all_frames(); signup_frame.pack(fill='both', expand=True);
        for w in signup_frame.winfo_children(): w.destroy();
        tk.Label(signup_frame, text="Create New Account", font=("Arial", 18, "bold"), bg="#fdf6ec").pack(pady=20)
        tk.Label(signup_frame, text="New Username", font=("Arial", 11), bg="#fdf6ec").pack(pady=(10,0)); new_username = tk.Entry(signup_frame, font=("Arial", 11), width=30); new_username.pack()
        tk.Label(signup_frame, text="New Password", font=("Arial", 11), bg="#fdf6ec").pack(pady=(10,0)); new_password = tk.Entry(signup_frame, show="*", font=("Arial", 11), width=30); new_password.pack()
        def register():
            username, password = new_username.get(), new_password.get()
            if not username or not password: messagebox.showerror("Error", "Please enter both fields"); return
            with open(USER_DATA_FILE, 'r') as f: users = json.load(f)
            if username in users: messagebox.showerror("Error", "Username already exists.")
            else:
                users[username] = password;
                with open(USER_DATA_FILE, 'w') as f: json.dump(users, f, indent=4)
                messagebox.showinfo("Success", "Account created. Please login."); show_login_signup()
        tk.Button(signup_frame, text="Create Account", command=register, bg="#007bff", fg="white", font=("Arial", 12, "bold")).pack(pady=20)
        tk.Button(signup_frame, text="⬅ Back to Login", command=show_login_signup, bg="#6c757d", fg="white").pack()
    tk.Button(login_frame, text="Login", command=login_action, bg="#28a745", fg="white", font=("Arial", 12, "bold")).pack(pady=20)
    tk.Button(login_frame, text="Create Account", command=show_signup_page, bg="#17a2b8", fg="white").pack(pady=5)
    tk.Button(login_frame, text="⬅ Back to Menu", command=show_menu, bg="#6c757d", fg="white").pack(pady=15)

def show_welcome_page():
    hide_all_frames(); welcome_frame.pack(fill='both', expand=True);
    for widget in welcome_frame.winfo_children(): widget.destroy();
    tk.Label(welcome_frame, text=f"Welcome, {current_user}!", font=("Georgia", 24, "bold"), bg="#e0f7fa", fg="#004d40").pack(pady=(50, 30))
    button_style = {"width": 20, "font": ("Arial", 14), "pady": 10}
    tk.Button(welcome_frame, text="Predict Parkinson's", command=show_predict_page, bg="#00796b", fg="white", **button_style).pack(pady=15)
    tk.Button(welcome_frame, text="Logout", command=logout, bg="#d32f2f", fg="white", **button_style).pack(pady=15)

def logout(): global current_user; current_user = None; messagebox.showinfo("Logout", "You have been successfully logged out."); show_menu()

def show_predict_page():
    global prediction_entries
    hide_all_frames(); predict_frame.pack(fill='both', expand=True);
    for widget in predict_frame.winfo_children(): widget.destroy();
    tk.Label(predict_frame, text="Enter Voice Measurements", font=("Georgia", 20, "bold"), bg="#fffde7", fg="#333").pack(pady=10)
    if not model_loaded: tk.Label(predict_frame, text="Prediction Model Not Found!\nPlease run train_model.py to enable this feature.", font=("Arial", 12, "bold"), bg="red", fg="white", pady=10).pack(pady=10)
    form_frame = tk.Frame(predict_frame, bg="#fffde7", padx=20, pady=10); form_frame.pack()
    prediction_entries = {}
    for i, feature in enumerate(FEATURES_FOR_MODEL):
        label = tk.Label(form_frame, text=f"{feature}:", font=("Arial", 11), bg="#fffde7"); label.grid(row=i, column=0, sticky="e", padx=5, pady=5)
        entry = tk.Entry(form_frame, font=("Arial", 11), width=15); entry.grid(row=i, column=1, sticky="w", padx=5, pady=5)
        prediction_entries[feature] = entry
    button_frame = tk.Frame(predict_frame, bg="#fffde7"); button_frame.pack(pady=15)
    predict_button_state = tk.NORMAL if model_loaded else tk.DISABLED
    tk.Button(button_frame, text="Get Prediction", command=perform_prediction, bg="#4caf50", fg="white", font=("Arial", 12, "bold"), state=predict_button_state).pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text="⬅ Back", command=show_welcome_page, bg="#6c757d", fg="white", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)

def perform_prediction():
    if not model_loaded: return
    try:
        input_values = [float(prediction_entries[feature].get()) for feature in FEATURES_FOR_MODEL]
        input_data_dict = dict(zip(FEATURES_FOR_MODEL, input_values))
        input_array = np.array(input_values).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)
        result_text = "Parkinson's Disease Detected" if prediction[0] == 1 else "No Disease Detected"
        is_positive = (prediction[0] == 1)
        show_result_page(result_text, is_positive, input_data_dict)
    except ValueError: messagebox.showerror("Input Error", "Please ensure all fields contain valid numbers.")
    except Exception as e: messagebox.showerror("Prediction Error", f"An unexpected error occurred: {e}")

def show_result_page(result_text, is_positive, input_data):
    hide_all_frames()
    result_frame.pack(fill='both', expand=True)
    for widget in result_frame.winfo_children(): widget.destroy()

    content_container = tk.Frame(result_frame, bg="#ffffff", padx=60, pady=40)
    content_container.place(in_=result_frame, anchor="c", relx=.5, rely=.5)

    TITLE_FONT = ("Arial", 22, "bold")
    BODY_FONT = ("Arial", 12)
    PREDICTION_FONT = ("Arial", 14, "bold")
    METRIC_FONT = ("Arial", 11)
    GREEN_COLOR = "#29b864"
    TEXT_COLOR = "#5a5a5a"
    RED_COLOR = "#d9534f" 
    
    header_frame = tk.Frame(content_container, bg="#ffffff")
    header_frame.pack(anchor='w', pady=(0, 25))
    tk.Label(header_frame, text="📊", font=("Arial", 26), bg="#ffffff", fg=GREEN_COLOR).pack(side=tk.LEFT)
    tk.Label(header_frame, text="Prediction Result", font=TITLE_FONT, bg="#ffffff", fg=GREEN_COLOR).pack(side=tk.LEFT, padx=10)

    inputs_to_show = ["MDVP:Fo(Hz)", "MDVP:Jitter(%)", "MDVP:Shimmer", "HNR", "PPE"]
    for feature in inputs_to_show:
        if feature in input_data:
            value = f"{input_data[feature]:.4f}" 
            tk.Label(content_container, text=f"{feature}: {value}", font=BODY_FONT, bg="#ffffff", fg=TEXT_COLOR).pack(anchor='w', pady=3)

    prediction_frame = tk.Frame(content_container, bg="#ffffff")
    prediction_frame.pack(anchor='w', pady=(25, 15))
    result_color = RED_COLOR if is_positive else GREEN_COLOR
    tk.Label(prediction_frame, text="🩺", font=("Arial", 22), bg="#ffffff", fg=result_color).pack(side=tk.LEFT)
    tk.Label(prediction_frame, text=f"Predicted Status: {result_text}", font=PREDICTION_FONT, bg="#ffffff", fg=result_color).pack(side=tk.LEFT, padx=10)

    metrics_frame = tk.Frame(content_container, bg="#ffffff")
    metrics_frame.pack(anchor='w', pady=10)
    
    accuracy = f"{model_performance.get('accuracy', 0.0):.2f}"
    precision = f"{model_performance.get('precision', 0.0):.2f}"
    recall = f"{model_performance.get('recall', 0.0):.2f}"
    f1_score = f"{model_performance.get('f1_score', 0.0):.2f}"
    
    tk.Label(metrics_frame, text=f"✅ Accuracy: {accuracy}", font=METRIC_FONT, bg="#ffffff", fg=TEXT_COLOR).grid(row=0, column=0, sticky='w', padx=(0, 40), pady=4)
    tk.Label(metrics_frame, text=f"📋 Recall: {recall}", font=METRIC_FONT, bg="#ffffff", fg=TEXT_COLOR).grid(row=1, column=0, sticky='w', padx=(0, 40), pady=4)
    tk.Label(metrics_frame, text=f"🎯 Precision: {precision}", font=METRIC_FONT, bg="#ffffff", fg=TEXT_COLOR).grid(row=0, column=1, sticky='w', padx=(0, 40), pady=4)
    tk.Label(metrics_frame, text=f"🧠 F1 Score: {f1_score}", font=METRIC_FONT, bg="#ffffff", fg=TEXT_COLOR).grid(row=1, column=1, sticky='w', padx=(0, 40), pady=4)
    
    button_container = tk.Frame(result_frame, bg="#f0f2f5")
    button_container.pack(side="bottom", fill="x", padx=50, pady=20)
    
    tk.Button(button_container, text="Predict Again", command=show_predict_page, bg="#e74c3c", fg="white", font=("Arial", 12, "bold"), relief="flat", width=15, height=2).pack(side="left")
    tk.Button(button_container, text="Logout", command=logout, bg="#3498db", fg="white", font=("Arial", 12, "bold"), relief="flat", width=15, height=2).pack(side="right")

# === THIS IS THE MODIFIED FUNCTION TO HANDLE BACKGROUND IMAGES ===
def create_info_page(frame, title, content_text, image_path=None):
    hide_all_frames()
    frame.pack(fill='both', expand=True)
    for w in frame.winfo_children(): w.destroy()

    # --- NEW: Background Image Logic ---
    if image_path and os.path.exists(image_path):
        try:
            from PIL import Image, ImageTk
            # Open the image and resize it to fit the window dimensions
            img = Image.open(image_path)
            # Use the main window's current size for a better fit
            resized_img = img.resize((root.winfo_width(), root.winfo_height()), Image.LANCZOS)
            bg_image = ImageTk.PhotoImage(resized_img)

            background_label = tk.Label(frame, image=bg_image)
            background_label.image = bg_image # IMPORTANT: Keep a reference!
            background_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            print(f"Error loading background image {image_path}: {e}")
            frame.config(bg="#f9f9f9") # Fallback to a solid color on error
    else:
        # If no image path is provided, use the default solid color
        frame.config(bg="#f9f9f9")

    # --- Widgets on top of the background ---
    # Using a semi-transparent frame for the title to improve readability
    title_frame = tk.Frame(frame, bg="#ffffff", relief="solid", bd=1)
    title_frame.pack(pady=(40, 20), padx=40)
    tk.Label(title_frame, text=title, font=("Georgia", 24, "bold"), bg="#ffffff", fg="#333").pack(padx=20, pady=10)

    # Content container with a semi-transparent white background
    content_container = tk.Frame(frame, bg="#ffffff", bd=1, relief="solid")
    content_container.pack(padx=40, pady=10, fill='both', expand=True)

    text_widget = scrolledtext.ScrolledText(content_container, wrap=tk.WORD, font=("Georgia", 13), 
                                            bg="#ffffff", fg="#222222", padx=15, pady=15, 
                                            relief="flat", borderwidth=0)
    text_widget.insert(tk.INSERT, content_text)
    text_widget.config(state=tk.DISABLED)
    text_widget.pack(padx=10, pady=10, fill='both', expand=True)
    
    tk.Button(frame, text="⬅ Back to Menu", command=show_menu, bg="#6c757d", fg="white", font=("Georgia", 12, "bold"), padx=20, pady=8).pack(pady=40)

# === THIS FUNCTION IS NOW MODIFIED TO CALL THE NEW create_info_page ===
def show_abstract():
    # Define the path for the background image
    bg_image_path = os.path.join(PROJECT_DIR, 'parkinsons_background.png')

    content = ("Parkinson’s Disease (PD) is a progressive neurological disorder that primarily affects movement control. "
               "This project focuses on the development of a machine learning-based system to detect PD using "
               "biomedical voice measurements. We employ a Support Vector Machine (SVM) classifier trained on a dataset of vocal "
               "features such as jitter, shimmer, and harmonics-to-noise ratio. The system provides a user-friendly interface "
               "for prediction and serves as a tool to aid in early diagnosis We employ supervised learning techniques, specifically a Support Vector Machine (SVM) classifier, to analyze features such as pitch, jitter, shimmer, and Harmonics-to-Noise Ratio (HNR). The dataset used for training and evaluation is sourced from the UCI Machine Learning Repository. Our model is trained to distinguish between healthy individuals and those affected by Parkinson’s Disease based on their vocal attributes.A user-friendly graphical interface built with Tkinter enables users to input new voice measurement data and receive instant predictions. The system achieves high accuracy and demonstrates the potential of machine learning as a valuable tool in medical diagnostics, especially for non-invasive, early-stage detection of Parkinson’s Disease.")
    
    # Call create_info_page and pass the image path
    create_info_page(abstract_frame, "Abstract", content, image_path=bg_image_path)


def show_algorithm(): 
    content = ("Algorithm: Support Vector Machine (SVM)\n\n"
               "An SVM is a supervised machine learning algorithm that works by finding a hyperplane that best separates data points into classes.\n\n"
               "1. Data Preprocessing: Features are scaled to a standard range to ensure equal contribution to the model.\n\n"
               "2. Training: The SVM is trained on the labeled dataset to learn the optimal boundary separating 'Healthy' from 'Parkinson's' cases.\n\n"
               "3. Prediction: The trained model classifies new voice data based on which side of the separating boundary it falls.\n\n"
               "Example: we developed a machine learning-based system to detect Parkinson’s Disease using vocal features. Parkinson’s is a neurological disorder that affects motor control and speech, and early detection can significantly improve patient outcomes. We used a dataset from the UCI Machine Learning Repository, which contains various biomedical voice measurements such as pitch, jitter, shimmer, and Harmonics-to-Noise Ratio (HNR). A Support Vector Machine (SVM) classifier was trained on this data to distinguish between healthy individuals and those with Parkinson’s Disease. The model achieved high accuracy in predictions, showing the effectiveness of using voice features for diagnosis. Additionally, we built a simple graphical user interface (GUI) using Tkinter in Python, which allows users to input voice-related values and receive real-time prediction results. This project demonstrates the potential of machine learning as a supportive tool in medical diagnosis, offering a non-invasive, cost-effective solution for early Parkinson’s detection. A user inputs voice metrics. These values form a feature vector, which the SVM model uses to predict a 'status' of 1 (PD) or 0 (Healthy).")
    # Call create_info_page without an image path to get the default look
    create_info_page(algorithm_frame, "Algorithm & Example", content)

def show_dataset_info():
    hide_all_frames(); dataset_info_frame.pack(fill='both', expand=True);
    for w in dataset_info_frame.winfo_children(): w.destroy();
    tk.Label(dataset_info_frame, text="Dataset Information", font=("Georgia", 20, "bold"), bg="#f9f9f9", fg="#333").pack(pady=10)
    container = tk.Frame(dataset_info_frame, bg="#ffffff", bd=2, relief="groove"); container.pack(padx=30, pady=5, fill='both', expand=True)
    content = ("\n🗞 Parkinson’s Disease Dataset Overview\n\n• Source: UCI Machine Learning Repository\n"
               "• Contains 195 voice recordings from 31 people (23 with PD, 8 healthy).\n"
               "• Purpose: Predict Parkinson’s based on vocal features.\n\n"
               "🧬 Key Features:\n1. MDVP:Fo(Hz) — Average vocal frequency\n2. MDVP:Jitter(%) — Pitch variation\n3. MDVP:Shimmer — Amplitude variation\n"
               "4. HNR — Harmonics-to-noise ratio\n5. PPE — Pitch Period Entropy\n"
               "6. status — Target: 1 (PD), 0 (Healthy)\n\n"
               "📂 Link:\nhttps://archive.ics.uci.edu/ml/datasets/parkinsons\n")
    text_widget = scrolledtext.ScrolledText(container, wrap=tk.WORD, font=("Georgia", 13), bg="#fefefe", fg="#222222", padx=15, pady=15, relief="flat", borderwidth=0)
    text_widget.insert(tk.INSERT, content); text_widget.config(state=tk.DISABLED); text_widget.pack(padx=10, pady=10, fill='both', expand=True)
    button_frame = tk.Frame(dataset_info_frame, bg="#f9f9f9"); button_frame.pack(pady=(10, 5))
    tk.Button(button_frame, text="📂 Open Data Folder", command=lambda: open_path(PROJECT_DIR), bg="#ffc107", fg="black", font=("Arial", 11, "bold"), padx=15, pady=5).pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text="📄 View Dataset File", command=lambda: open_path(DATASET_FILE_PATH), bg="#17a2b8", fg="white", font=("Arial", 11, "bold"), padx=15, pady=5).pack(side=tk.LEFT, padx=10)
    tk.Button(dataset_info_frame, text="⬅ Back to Menu", command=show_menu, bg="#6c757d", fg="white", font=("Georgia", 12, "bold"), padx=20, pady=8).pack(pady=(5, 15))

def show_help():
    content = ("Need Assistance?\n\n"
               "• How to Use: \n"
               "1. From the Home Page, click 'Proceed' to go to the Main Menu.\n"
               "2. To use the prediction tool, select 'Login / Sign Up' and log into your account.\n"
               "3. Once logged in, click 'Predict Parkinson's'.\n"
               "4. Fill in all the voice measurement fields and click 'Get Prediction'.\n\n"
               "• Troubleshooting:\n"
               "If the 'Get Prediction' button is disabled, it means the model files (`svm_model.joblib`, `scaler.joblib`) were not found. Please run the `train_model.py` script to generate them.\n"
               "If the performance scores are missing on the result page, ensure 'model_performance.json' was created by the training script.")
    # Call create_info_page without an image path to get the default look
    create_info_page(help_frame, "Help and Support", content)

if __name__ == "__main__":
    show_home()
    root.mainloop()