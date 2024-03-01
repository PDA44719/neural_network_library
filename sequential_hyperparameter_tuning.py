import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from sklearn.model_selection import KFold

class Regressor(nn.Module):

    def __init__(self, x, nb_epoch=800, batch_size=64, learning_rate=0.001, loss_fn=nn.MSELoss(), model=None):
      
        super(Regressor, self).__init__()
        
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        
        # Initialize the preprocessor with the training data
        self.preprocessor, self.input_size = self._initialise_preprocessor(x)
        
        self.output_size = 1

        # Define the model architecture
        if model is None:
            self.model = nn.Sequential(
                nn.Linear(self.input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 30),
                nn.ReLU(),
                nn.Linear(30, 10),
                nn.ReLU(),
                nn.Linear(10, self.output_size),
            )
        else:
            self.model = model
            

        self.optimiser = optim.Adam(self.model.parameters(), lr=self.learning_rate)


    def _initialise_preprocessor(self, x):
        numerical_cols = x.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = x.select_dtypes(include=['object']).columns

        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Filling missing values with median
            ('scaler', StandardScaler()),  # Standardizing the numerical features
        ])

        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handling missing categories
            ('onehot', OneHotEncoder(handle_unknown='ignore')),  # Encoding categorical features
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols),
        ])

        preprocessed_x = preprocessor.fit_transform(x)
        input_size = preprocessed_x.shape[1]

        return preprocessor, input_size


    def _preprocessor(self, x, y = None, training = False):
       
        # Apply the preprocessing to the dataset
        if training:
            preprocessed_data = self.preprocessor.fit_transform(x)
        else:
            preprocessed_data = self.preprocessor.transform(x)
        preprocessed_data = torch.tensor(preprocessed_data, dtype=torch.float32)
        if y is not None:
            y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        
        return preprocessed_data, y

    
    def fit(self, x, y):

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

        X_train, _ = self._preprocessor(x_train)
        Y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        X_val, _ = self._preprocessor(x_val)
        Y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

        validation_rmse = []  # Initialize outside the loop

        for epoch in range(self.nb_epoch):
            self.model.train()
            for inputs, targets in train_loader:
                self.optimiser.zero_grad()
                predictions = self.model(inputs)
                loss = self.loss_fn(predictions, targets)
                loss.backward()
                self.optimiser.step()

            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val)
                val_loss = self.loss_fn(val_predictions, Y_val).item()
            
            epoch_rmse = np.sqrt(val_loss)
            validation_rmse.append(epoch_rmse)
        
            print(f'Epoch {epoch+1}: Validation RMSE = {epoch_rmse}')
            
        return self, validation_rmse
        
            
    def predict(self, x):
      

        X, _ = self._preprocessor(x, training=False)  # Preprocess inputs
        
        self.model.eval()  # Set the model to evaluation mode
        
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.numpy()


    def score(self, x, y):
      

        X, _ = self._preprocessor(x, training=False)  # Preprocess inputs
        Y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)  # Ensure Y is the correct shape
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        
        mse = mean_squared_error(y.values, predictions.numpy())
        rmse = np.sqrt(mse)  # Compute the RMSE from MSE
        return rmse



def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model

def perform_hyperparameter_search(x, y): 
    batch_sizes = [16, 32, 64]
    fixed_learning_rate = 0.01
    fixed_epochs = 800
    rmse_scores = {batch_size: [] for batch_size in batch_sizes}

    for batch_size in batch_sizes:
        print(f"Evaluating model with batch size: {batch_size}")
        regressor = Regressor(x, nb_epoch=fixed_epochs, batch_size=batch_size, learning_rate=fixed_learning_rate)
        _, validation_rmse = regressor.fit(x, y)
        rmse_scores[batch_size] = validation_rmse

    # Plotting the RMSE scores for each batch size from the 10th epoch onwards
    plt.figure(figsize=(10, 6))
    for batch_size, scores in rmse_scores.items():
        plt.plot(range(10, fixed_epochs + 1), scores[9:], label=f'Batch Size={batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation RMSE')
    plt.title('Validation RMSE per Epoch for Different Batch Sizes')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Determine the best batch size based on final RMSE value
    final_rmse_scores = {batch_size: scores[-1] for batch_size, scores in rmse_scores.items()}
    best_batch_size = min(final_rmse_scores, key=final_rmse_scores.get)
    print(f"Best batch size: {best_batch_size} with RMSE: {final_rmse_scores[best_batch_size]}")

    # Train the final model with the best batch size
    best_model = Regressor(x, nb_epoch=fixed_epochs, batch_size=best_batch_size, learning_rate=fixed_learning_rate)
    best_model.fit(x, y)

    return best_model, best_batch_size

'''
def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    best_hyperparams, best_model = perform_hyperparameter_search(x, y)
    print(f"Best Hyperparameters: {best_hyperparams}")
    save_regressor(best_model)

    # Error
    error = best_model.score(x, y)
    print(f"\nRegressor error on the training set: {error}\n")


if __name__ == "__main__":
    example_main()
'''

def cross_validate_model(x, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = []

    for train_index, val_index in kf.split(x):
        x_train, x_val = x.iloc[train_index], x.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = Regressor(x_train, nb_epoch=800, batch_size=64, learning_rate=0.01)
        model.fit(x_train, y_train)

        _, validation_rmse = model.fit(x_train, y_train)  # Fit the model
        model_rmse = model.score(x_val, y_val)  # Evaluate the model
        rmse_scores.append(model_rmse)

    average_rmse = np.mean(rmse_scores)
    print(f"Average RMSE across all folds: {average_rmse}")
    return average_rmse


def main():
    
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 
    # Prepare the features (X) and target (y)
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Ensure that the target is in the correct format (might not be necessary depending on your dataset)
    y = y.astype(float)

    # Perform 10-fold cross-validation
    average_rmse = cross_validate_model(x, y)

    print(f"Average RMSE from 10-fold cross-validation: {average_rmse}")

# Ensure to define the cross_validate_model function before this
if __name__ == "__main__":
    main()
