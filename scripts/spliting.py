import pandas as pd


class SplitData:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def test_validate_train(self,train_size:float,valid_size:float,y_value:str) -> pd.DataFrame:
        df=self.df
        
        if train_size+valid_size>1:
            raise ValueError("train_size and valid_size must be less than 1")

        train_index = int(len(df) * train_size)

        df.sort_values(by=y_value, ascending=True, inplace=True)

        df_train = df[0:train_index]
        df_rem = df[train_index:]

        valid_index = int(len(df) * valid_size)

        df_valid = df[train_index : train_index + valid_index]
        df_test = df[train_index + valid_index :]

        X_train, y_train = (
            df_train.drop(columns=y_value).copy(),
            df_train[y_value].copy(),
        )
        X_valid, y_valid = (
            df_valid.drop(columns=y_value).copy(),
            df_valid[y_value].copy(),
        )
        X_test, y_test = df_test.drop(columns=y_value).copy(), df_test[y_value].copy()

        print(X_train.shape), print(y_train.shape)
        print(X_valid.shape), print(y_valid.shape)
        print(X_test.shape), print(y_test.shape)
        
        return {
            "x_train": X_train,
            "y_train": y_train,
            "x_valid": X_valid,
            "y_valid": y_valid,
            "x_test": X_test,
            "y_test": y_test
        }
        