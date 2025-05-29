import pandas as pd
from sklearn.datasets import load_iris
from pathlib import Path

# Check if data folder exists
Path("data").mkdir(parents=True, exist_ok=True)

iris = load_iris(as_frame=True)
df = iris.frame
df.columns = df.columns.str.replace(" ", "_")
df.to_csv("data/iris.csv", index=False)
print("Iris dataset generado correctamente")

