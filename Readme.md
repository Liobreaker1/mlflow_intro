# MLflow Intro Tutorial

This project shows a basic example of model tracking with MLflow and the Iris dataset.

---

## Project Structure

```
mlflow-intro-tutorial/
├── data/
│   └── iris.csv
├── scripts/
│   └── generate_iris.py
├── src/
│   └── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Creating the virtual environment

```bash
python3.11 -m venv mlflow_env
```

### 2. Activate virtual environment

```bash
# macOS/Linux
source mlflow_env/bin/activate

# Windows
mlflow_env\Scripts\activate
```

### 3. Installing dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---


## Setting up Virtaul environment in VSCode.

```bash
pip install ipykernel
```

### Register the environment in local

```bash
python -m ipykernel install --user --name=mlflow-env --display-name "Python3.11 (mlflow-env)"
```

Now the environment should appear like:

```
Python3.11 (mlflow-env)
```


---

## Loading the Iris dataset

Executing the following scrpt will help downloading the Iris dataset into the `data` folder.

```python
# Run just once
import pandas as pd
from sklearn.datasets import load_iris
df = load_iris(as_frame=True).frame
df.to_csv("data/iris.csv", index=False)
```

---

## Training and tracking the model

Executing the training file:

```bash
python src/train_model.py
```

The previous chunk will:
- Train the model `RandomForestClassifier`
- Register:
  - Params: `n_estimators`
  - Méetrics: `accuracy`
  - Serialized model
  - Imput example (`input_example`)

---

## Opening MLflow UI to visualize data.

1. Running MLflow service

```bash
mlflow ui
```

2. Open the browser

```
http://127.0.0.1:5000
```

---

## To make Model Registry available locally

```
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```