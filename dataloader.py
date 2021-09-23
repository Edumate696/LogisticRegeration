import pandas as pd
from sklearn import datasets


class DataLoader:
    __data = datasets.load_breast_cancer(as_frame=True).frame

    @staticmethod
    def load_dataset() -> pd.DataFrame:
        return DataLoader.__data


if __name__ == '__main__':
    df = DataLoader.load_dataset()

    print('DataSet Example ->')
    print(df.head(10))

    print()
    print('Target Names :', datasets.load_breast_cancer().target_names)

    print()
