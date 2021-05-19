# multipleRegression
Lightweight Multiple Regression model with t-test functionality.

Example usage:

```python
import pandas as pd

from model import Model

df = pd.read_csv("room.csv")
Y = df["Price"]
X = df.drop(columns=["Price"])
m = Model(Y, X)

m.regression()
```
