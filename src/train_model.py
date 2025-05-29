import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Loading the dataset
df = pd.read_csv("data/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
n_estimators = 150
model = RandomForestClassifier(n_estimators=n_estimators)
model.fit(X_train, y_train)

# Model evaluation
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# MLflow Tracking
with mlflow.start_run():
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", acc)
    
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_train.iloc[:5])

    print(f"Accuracy logged: {acc:.4f}")
