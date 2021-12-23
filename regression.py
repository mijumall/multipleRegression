import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Model():
    def __init__(self, Y, X, category=[]):
        """Multiple regression.

        :type Y: class "pandas.core.series.Series" or "pandas.core.frame.DataFrame"
        :param Y: Dependent (explained) variable
        
        :type X: class "pandas.core.series.Series" or "pandas.core.frame.DataFrame"
        :param X: Independent (explaining) variable(s). 
        
        :type category: list[str]
        :param category: 
            This parameter should contain the names of columns from which you wish to convert 
            to dummy variables. Parameter X must have the same columns too.
            
            Example:
            
            category = ['day']
            
            column 'day' contains the following categorical values.
            ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
            
            If category list is passed, the columns in the category list are replaced with newly created 
            dummy variables based on categorical values.
            
            New columns:
            ['day_Mon', 'day_Tue', 'day_Wed', 'day_Thu', 'day_Fri']
        """
        self._init_normal_distribution()
        self.N = len(Y)
        keys = pd.Series(range(0, self.N))
        
        # Reassign new index in case Y or X are coming from slices, which causes "matrices not aligned error"
        self.Y = pd.Series(pd.DataFrame(Y).set_index(keys=keys).iloc[:,0])
        self.X = pd.DataFrame(X).set_index(keys=keys)
        
        # Convert categorical values to dummy variables
        if len(category) == 0:
            pass
        else:
            for d in category:
                self.categorical2dummy(d)

    def regression(self, showCorrelation=True):
        """Compute and print multiple regression results.
        
        :type showCorrelation: bool
        :param showCorrelation: 
            Set False if you don't need to check multicollinearity, 
            defaults to True.
        """
        
        print("Regression starts... \n")
        
        Y = self.Y
        X = self.X
        k = len(self.X.columns) # Number of independent variables
        N = self.N
        
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
        
        # V_hat shapes 3 x 3 symmetric matrix. To get V for each coefficient, extract diagonal matrix:
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
        
        print(f"Explained variable: {self.Y.name}\n")
        print(f"Adjusted R-squared: {round(R2, 4)}\n")
        print("Two-tailed t-test results:\n")
        print(df, '\n')
        
        if showCorrelation:
            print("\nCorrelation between independent variables:\n")
            print(X.drop(columns="_constant_").corr())
        
    
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
            plt.figure(figsize=(24, 6))
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            
            def plot(title, place, x, y):
                plt.subplot(place)
                plt.title(title, fontsize=30)
                plt.plot(x, y)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.grid(alpha=0.5)
                
            plot("SND", 121, self.sndx, self.pdf / n )
            plot("CDF", 122, self.sndx, self.cdf)
            
            plt.show()
            
    def categorical2dummy(self, category):
        '''Create new DataFrame-shaped independent variables (self.X) with newly created dummy columns,
        which is to be generated from categorical values from givem category.
        
        :type category: str
        :param category: The column name
        '''
        
        df = self.X
        
        ### Below code creates the map of each categorical value and new column name. 
        ###
        ### Example:
        ###
        ### category: 'day'
        ### each value: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        ### 
        ### In this case, category_map would be like this:
        ###
        ### categories_map = {
        ###     'Mon': 'day_Mon',
        ###     'Tue': 'day_Tue',
        ###     'Wed': 'day_Wed',
        ###     ...
        ### }
        
        categories = list(df[df[category].duplicated() == False][category])
        categories_map = {c: f'{category}_{c}' for c in categories}
        
        # For columns, replace current 'category' with values of 'categories_map'
        current_cols = list(df.columns)
        category_index = current_cols.index(category)
        new_added_cols = [categories_map[c] for c in categories[1:]] # One of dummy variables must be dropped
        
        if category_index == len(current_cols):
            new_cols = current_cols[:category_index] + new_added_cols
        else:
            new_cols = current_cols[:category_index] + new_added_cols + current_cols[category_index + 1:]
            
        # Create new columns (dummy variables) in df. All values are set to 0 initially.
        for c in categories_map:
            df[categories_map[c]] = 0
            
        # Set each dummy value to 1 based on categories in 'dummy' column
        for idx, each_row in df.iterrows():
            c = each_row[category]
            df.at[idx, categories_map[c]] = 1
            
            
        self.X = df[new_cols]
        
