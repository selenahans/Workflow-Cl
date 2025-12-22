import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Eksperimen_SML_Selena")
# mlflow.set_tracking_uri("https://dagshub.com/selenahans/Eksperimen-SML-Selena.mlflow")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Student_Performance_Preprocessed.csv")
df = pd.read_csv(DATA_PATH)

df = df.fillna(0)

X = df.drop(columns=["Performance Index"])
y = df["Performance Index"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():

   
    mlflow.sklearn.autolog()

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2_Score", r2)

    residuals = y_test - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted Value")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.savefig("residual_plot.png")
    plt.close()

    mlflow.log_artifact("residual_plot.png")

    with open("regression_metrics.txt", "w") as f:
        f.write(f"MAE  : {mae}\n")
        f.write(f"RMSE : {rmse}\n")
        f.write(f"R2   : {r2}\n")

    mlflow.log_artifact("regression_metrics.txt")


    mlflow.sklearn.log_model(model, "random_forest_regression_model")

print("Training selesai dan tercatat di MLflow (DagsHub)")
