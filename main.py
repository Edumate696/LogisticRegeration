import numpy as np
from model import LogisticRegression

from dataloader import DataLoader


df = DataLoader.load_dataset()

X = df.values[:, :-1].T
y = df.values[:, -1:].T

model = LogisticRegression()

model.fit(X, y)



