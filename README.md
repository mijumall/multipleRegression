# multipleRegression
Lightweight Multiple Regression model with t-test functionality.
The standard errors are robust and appropriate under both heteroskedasticity and homoskedasticity.


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

The output would look like the following:
```
Regression starts... 

Adjusted R-squared: 0.7525

Two-tailed t-test results:

         Name          Coef      Std Err    t-value       p-value
0  _constant_  45522.952887  4301.355214  10.583398  0.000000e+00
1       Large   2402.201077   186.803291  12.859522  0.000000e+00
2         Old   -892.453408   181.975312  -4.904255  3.451671e-07
```

