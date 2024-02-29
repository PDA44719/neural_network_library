from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

class Regressor(nn.Module):
    def __init__(self, x, nb_epoch=400, batch_size=16, learning_rate=0.001, loss_fn=nn.MSELoss(), model=None):
        super(Regressor, self).__init__()
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        
        # Initialize Preprocessor
        self.preprocessor = self._initialize_preprocessor(x)
        self.y_preprocessor = StandardScaler()
        
        # Dummy preprocessing to determine input size
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        
        # Define the model architecture
        if model is None:
            self.model = nn.Sequential(
                nn.Linear(self.input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.output_size),
                # nn.ReLU(), # No activation function for the output layer
            )
        else:
            self.model = model
        
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _initialize_preprocessor(self, x):
        numerical_cols = x.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = x.select_dtypes(include=['object', 'category']).columns

        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])

        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols),
        ])

        return preprocessor

    def _preprocessor(self, x, y=None, training=False):
        if training:
            preprocessed_data = self.preprocessor.fit_transform(x)
            if y is not None:
                preprocessed_y = self.y_preprocessor.fit_transform(y.values.reshape(-1, 1))
        else:
            preprocessed_data = self.preprocessor.transform(x)
            if y is not None:
                preprocessed_y = self.y_preprocessor.transform(y.values.reshape(-1, 1))

        preprocessed_data = torch.tensor(preprocessed_data, dtype=torch.float32)
        preprocessed_y = torch.tensor(preprocessed_y, dtype=torch.float32).flatten() if y is not None else None

        return preprocessed_data, preprocessed_y

    def fit(self, x, y):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
        X_train, Y_train = self._preprocessor(x_train, y=y_train, training=True)
        X_val, Y_val = self._preprocessor(x_val, y=y_val, training=False)
        
        dataset = DataLoader(TensorDataset(X_train, Y_train), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.nb_epoch):
            self.model.train()
            total_loss = 0
            for inputs, targets in dataset:
                self.optimiser.zero_grad()
                predictions = self.model(inputs)
                loss = self.loss_fn(predictions, targets)
                loss.backward()
                self.optimiser.step()
                total_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val)
                val_loss = self.loss_fn(val_predictions, Y_val).item()

            print(f'Epoch {epoch+1}, Training Loss: {total_loss / len(dataset)}, Validation Loss: {val_loss}')

        return self

    def predict(self, x):
        self.model.eval()
        X, _ = self._preprocessor(x, training=False)
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.numpy()

    def score(self, x, y):
        self.model.eval()
        X, Y = self._preprocessor(x, y=y, training=False)
        with torch.no_grad():
            predictions = self.model(X)
        mse = mean_squared_error(Y.numpy(), predictions.numpy())
        return mse

def example_main():
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")

    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    regressor = Regressor(x_train, nb_epoch=1000, batch_size=32, learning_rate=0.001)
    regressor.fit(x_train, y_train)

    error = regressor.score(x_train, y_train)
    print(f"\nRegressor error: {error}\n")

if __name__ == "__main__":
    example_main()
