import os
import pandas as pd

HISTORY_FILE = "data/prediction_history.csv"

def save_prediction(input_data, prediction):

    row = input_data.copy()
    row["Prediction"] = prediction

    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    else:
        history = pd.DataFrame([row])

    history.to_csv(HISTORY_FILE, index=False)

    print("Prediction History Saved Successfully")