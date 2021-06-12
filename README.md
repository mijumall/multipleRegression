# multipleRegression
Simple Multiple Regression model with t-test functionality.
The standard errors are robust and appropriate under both __heteroskedasticity__ and __homoskedasticity__.


Example usage:

```python
import pandas as pd

from regression import Model

df = pd.read_csv("room.csv")
Y = df["Price"]
X = df.drop(columns=["Price"])

m = Model(Y, X)
m.regression(showCorrelation=False) # Turn it True if you wish to check multicollinearity
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

If you're using Linux (Ubuntu, CentOS) and encounter memory error while working with large datasets, try using swap memory. Here's how to create it:
```bash
# Check how much memory are available right now.
free -h 

# Create a directry to put your swapfile on.
sudo mkdir /swap

# If you want to create additional 4096MB (4GB) memory for example, run the following:
sudo dd if=/dev/zero of=/swap/swapfile bs=1M count=4096 status=progress
sudo chmod 600 /swap/swapfile
sudo mkswap /swap/swapfile
sudo swapon /swap/swapfile

# Confirm you've got additional swap memory
free -h 
```


If you don't need swap memory anymore, you can disable it and delete it.
```bash
# Disable swapmemory
sudo swapoff /swap/swapfile

# Delete swapfile
sudo rm /swap/swapfile
```
