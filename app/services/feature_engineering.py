import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Keep numeric columns only
    num_df = df.select_dtypes(include="number")
    # Fill NaNs with column mean
    num_df = num_df.fillna(num_df.mean(numeric_only=True))
    return num_df
