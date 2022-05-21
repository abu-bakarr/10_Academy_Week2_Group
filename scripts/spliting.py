import pandas as pd
from sklearn.model_selection import train_test_split


class SplitData:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def test_validate_train(self,train_size:float,y_value:str) -> pd.DataFrame:
        df=self.df
        
        if train_size>1:
            raise ValueError("train_size must be less than 1")

        
        X1 = df.drop(y_value, axis=1) #prediction features
        y1 = df[y_value]
        
        X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.1, random_state=42
)
        return X1_train, X1_test, y1_train, y1_test;