# multipleRegression
Lightweight Multiple Regression model with t-test functionality.

Example usage:

```python
from model import Model, load_dataset

df = load_dataset()
Y = df["Price"]
X = df.drop(columns=["Price"])
m = Model(Y, X)

m.regression()
```