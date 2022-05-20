# Required imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Function for plotting Univariate graphs
def plot(df:pd.DataFrame, x_col:str, title:str,rotation=0):
    plt.figure(figsize=(12, 6))
    plt.title(title);
    plt.xticks(rotation=rotation)
    sns.countplot(data=df, x=x_col)
  
    plt.show()

# Function for importing Bivariate graphs
    # Plot function
def bi_plot(df:pd.DataFrame, x_col:str, y_col:str, title:str, rotation=0):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x=x_col , hue = y_col)
    plt.title(title)
    plt.xticks(rotation=rotation)
    
    plt.show()