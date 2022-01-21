# Multiple Regression
Simple Multiple Regression model with t-test functionality.
The standard errors are robust and appropriate under both heteroskedasticity and homoskedasticity.


Example usage:

```python
import pandas as pd

from regression import Model 

df = pd.read_csv('room.csv')

Y = df['Price']
X = df[['Large', 'Old', 'Orientation']]

m = Model(Y, X, category=['Orientation'])
m.regression(correlation=False)
```

The output would look like the following:
```
Regression starts... 

Explained variable: Price

Adjusted R-squared: 0.7586

Two-tailed t-test results:

                Name          Coef      Std Err    t-value   p-value
0         _constant_  48358.021215  4731.727837  10.219950  0.000000
1              Large   2249.490178   196.347455  11.456681  0.000000
2                Old   -809.880972   181.618833  -4.459235  0.000008
3  Orientation_North   -749.426764  2147.045222  -0.349050  0.724785
4   Orientation_West  -1627.540061  2087.854230  -0.779528  0.432948
5  Orientation_South   4311.759920  2865.106092   1.504922  0.131477
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
