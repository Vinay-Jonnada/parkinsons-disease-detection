import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json

# === Configuration ===
# This script trains the model and saves all necessary files for the GUI.
# It must be run successfully before the main GUI application will work correctly.
PROJECT_DIR = 'D:/parkinsons_project'
DATASET_FILE = os.path.join(PROJECT_DIR, 'parkinsons.csv')
MODEL_FILE = os.path.join(PROJECT_DIR, 'svm_model.joblib')
SCALER_FILE = os.path.join(PROJECT_DIR, 'scaler.joblib')
PERFORMANCE_FILE = os.path.join(PROJECT_DIR, 'model_performance.json')

# These features must be the EXACT same as in the GUI file.
FEATURES_FOR_MODEL = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", 
    "MDVP:Shimmer", "Shimmer:DDA", "HNR", "RPDE", "DFA", 
    "spread1", "spread2", "D2", "PPE"
]

def train_and_save_model():
    """
    Loads the dataset, trains an SVM model, evaluates it, and saves the 
    model, scaler, and a JSON file with performance metrics.
    """
    print("--- Starting Model Training Process ---")

    # 1. Load Data
    print(f"\n[STEP 1] Loading data from: {DATASET_FILE}")
    try:
        df = pd.read_csv(DATASET_FILE)
        if df.empty:
            print("Error: The CSV file is empty.")
            return
        print("-> Successfully loaded dataset.")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{DATASET_FILE}'")
        print("Please make sure 'parkinsons.csv' is in the 'D:/parkinsons_project' directory.")
        return

    # 2. Prepare Data
    print("\n[STEP 2] Preparing data for training...")
    # The target variable is 'status' (1 for Parkinson's, 0 for healthy)
    X = df[FEATURES_FOR_MODEL]
    y = df['status']
    print(f"-> Features selected: {len(FEATURES_FOR_MODEL)}")
    print(f"-> Target variable: 'status'")

    # 3. Split Data into Training and Testing sets
    print("\n[STEP 3] Splitting data into training and testing sets...")
    # stratify=y ensures the proportion of classes is the same in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"-> Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # 4. Scale Features
    print("\n[STEP 4] Scaling features using StandardScaler...")
    # SVMs are sensitive to feature scaling, so this is a crucial step.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("-> Features have been successfully scaled.")

    # 5. Train the SVM Model
    print("\n[STEP 5] Training the Support Vector Machine (SVC) model...")
    # Using probability=True can be useful but makes training slower. It's kept here.
    model = SVC(kernel='linear', C=10, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    print("-> Model training is complete.")

    # 6. Evaluate the Model and Extract Metrics
    print("\n[STEP 6] Evaluating the model and extracting performance metrics...")
    y_pred = model.predict(X_test_scaled)
    
    # This is the key step to get metrics as a dictionary
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    # We get overall accuracy from the report itself for consistency
    accuracy = report_dict['accuracy']
    # We extract metrics for the positive class ('1' for Parkinson's)
    # The .get() method is used to avoid errors if the class is not in the test set
    precision = report_dict.get('1', {}).get('precision', 0.0)
    recall = report_dict.get('1', {}).get('recall', 0.0)
    f1_score = report_dict.get('1', {}).get('f1-score', 0.0)

    # Create the final dictionary to be saved as JSON
    performance_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }
    
    print("\n--- Model Evaluation Results ---")
    print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Parkinsons (1)']))
    print(f"Extracted Metrics for GUI: {performance_metrics}")

    # 7. Save the Model, Scaler, and Performance Metrics
    print("\n[STEP 7] Saving model, scaler, and performance file...")
    # Ensure the project directory exists
    os.makedirs(PROJECT_DIR, exist_ok=True)
    
    # Save the trained model and scaler
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    
    # Save the performance metrics to a JSON file
    # The GUI will read this file to display the scores.
    with open(PERFORMANCE_FILE, 'w') as f:
        json.dump(performance_metrics, f, indent=4)
        
    print(f"-> Successfully saved model to: {MODEL_FILE}")
    print(f"-> Successfully saved scaler to: {SCALER_FILE}")
    print(f"-> Successfully saved performance metrics to: {PERFORMANCE_FILE}")
    print("\n--- Training and Saving Process Finished Successfully! ---")
    print("You can now run the main GUI application.")

if __name__ == '__main__':
    train_and_save_model()