import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Model():
    def __init__(self, Y, X):
        """Multiple regression.

        :param Y: Dependent (explained) variable
        :type Y: class "pandas.core.series.Series"
        
        :param X: Independent (eplaining) variable(s). 
        :type X: class "pandas.core.series.Series" or "pandas.core.frame.DataFrame"
        
        Example: 
            import pandas as pd
            
            from model import Model
            
            df = pd.read_csv("room.csv")
            Y = df["Price"]
            X = df.drop(columns=["Price"]) 
            m = Model(Y, X)
        """
        self._init_normal_distribution()
        self.Y = Y
        self.X = pd.DataFrame(X)
        self.N = len(Y)
        
    def regression(self):
        """Compute and print multiple regression results."""
        
        print("Regression starts... \n")
        
        Y = self.Y
        X = self.X
        k = len(self.X.columns) # Number of explaining variables
        N = len(self.Y)
        
        # Add constant vector and rearrange X's order to compute coefficient.
        X["_constant_"] = 1
        cols=list(X.columns)
        cols = [cols[-1]] + cols[:-1]
        X = X[cols]
        self.X = X 

        # Compute multiple regression.
        coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y
        
        # Compute and generate Y_hat as: pandas.core.series.Series
        Y_hat = {}
        for coef, column in zip(coefficients, X.columns):
            Y_hat[column] = coef * X[column]
        Y_hat = pd.DataFrame(Y_hat)
        Y_hat = Y_hat.T.sum()
        
        # Compute resudual and Adjusted R squared.
        residual = Y - Y_hat
        R2 = 1 - ((N-1) / (N-k-1)) * ((residual**2).sum() / ((Y - Y.mean())**2).sum())
        
        # Compute V_hat
        V_hat = (np.linalg.inv((1/N * X.T @ X))) @ \
                    (1/(N-k-1) * X.T @ np.diag(residual**2) @ X) @ \
                    (np.linalg.inv((1/N) * X.T @ X))
        # V_hat shapes 3 x 3 symmetric matrix. To get V for each coefficient, sum them up:
        V_hat = np.diag(V_hat)
        
        # Standard error of coefficients
        stdErr = ((1/N) * V_hat) ** (1/2)
        
        # t-value
        t_values = coefficients / stdErr
        
        # p-value
        n = len(self.pdf)
        p_values = []
        for t_value in t_values:
            p_value = 0.0
            for idx in range(n//2, -1, -1):
                if abs(t_value) < abs(self.sndx[idx]):
                    p_value = self.cdf[idx-1] * 2
                    break
            p_values.append(p_value)
        
        # Clearly visualize the data
        df = pd.DataFrame(
            {
                "Name": X.columns,
                "Coef": coefficients, 
                "Std Err": stdErr, 
                "t-value": t_values, 
                "p-value": p_values, 
            }
        )
        
        print(f"Adjusted R-squared: {round(R2, 4)}\n")
        print("Two-tailed t-test results:\n")
        print(df)
        
    
    def _init_normal_distribution(self):
        """Compute the following values and make them be available as class instances.
        This method is called in __init__ function.
        
        self.sndx   # Stands for: Standard Normal Distribution's x
        self.cdf    # Stands for: Cummulative Density Function
        self.pdf    # Stands for: Probability Density Function
        """
        n = 5000
        mu = 0
        sigma = 1
        
        x = np.linspace(-5, 5, n)
        
        pdf = (np.exp((-(x - mu) ** 2) / 2 * (sigma ** 2))) / (sigma * (1/2 ** (2 * np.pi)))
        
        cdf = []
        for idx in np.arange(n):
            cdf.append((pdf[:idx] / pdf.sum()).sum())
        cdf = np.array(cdf)

        self.sndx = x 
        self.cdf = cdf
        self.pdf = pdf
        
    def visualizeDistributions(self):
        n = len(self.pdf)
        with plt.style.context("dark_background"):
            plt.figure(figsize=(22, 9))
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            
            def plot(title, place, x, y):
                plt.subplot(place)
                plt.title(title, fontsize=30)
                plt.plot(x, y)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.grid(alpha=0.5)
                
            plot("SND", 221, self.sndx, self.pdf / n )
            plot("CDF", 222, self.sndx, self.cdf)
            plot("SND (n)", 223, list(range(n)), self.pdf / n)
            plot("CDF (n)", 224, list(range(n)), self.cdf)
            
            plt.show()

