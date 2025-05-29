import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Loading the dataset
df = pd.read_csv("data/iris.csv")
X = df.drop("target", axis=1)
X += np.random.normal(0, 0.1, X.shape)
y = df["target"]

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Model training
n_estimators = 250
model = RandomForestClassifier(n_estimators=n_estimators)
# model = KNeighborsClassifier(n_neighbors=3)
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

    # Register model in Model Registry

    import time
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    result = mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
        name="IrisClassifier"
    )

    for _ in range(10):
        model_info = client.get_model_version(name="IrisClassifier", version=result.version)
        if model_info.status == "READY":
            break
        time.sleep(1)

    print(f"Registered Model: IrisClassifier v{result.version}")   
