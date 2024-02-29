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
import matplotlib.pyplot as plt

class Regressor(nn.Module):

    def __init__(self, x, nb_epoch=1000, batch_size=16, learning_rate=0.001, loss_fn=nn.MSELoss(), model=None):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

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
                # nn.ReLU(),
            )
        else:
            self.model = model
            

        self.optimiser = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

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
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Apply the preprocessing to the dataset
        preprocessed_y = None
        if training:
            preprocessed_data = self.preprocessor.fit_transform(x)
            if y is not None:
                preprocessed_y = self.y_preprocessor.fit_transform(y)
        else:
            preprocessed_data = self.preprocessor.transform(x)
            if y is not None:
                preprocessed_y = self.y_preprocessor.transform(y)

        preprocessed_data = torch.tensor(preprocessed_data, dtype=torch.float32)
        if preprocessed_y is not None:
            preprocessed_y = torch.tensor(preprocessed_y, dtype=torch.float32)
        
        return preprocessed_data, preprocessed_y


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    
        X_train, _ = self._preprocessor(x_train)
        Y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        X_val, _ = self._preprocessor(x_val)
        Y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

        training_rmse = []
        validation_rmse = []
        start_epoch = 10  # Epoch from which to start recording the RMSE

        for epoch in range(self.nb_epoch):
            self.model.train()
            total_loss = 0
            for inputs, targets in train_loader:
                self.optimiser.zero_grad()
                predictions = self.model(inputs)
                loss = self.loss_fn(predictions, targets)
                loss.backward()
                self.optimiser.step()
                total_loss += loss.item()

            if epoch >= start_epoch:
                # Compute RMSE for training and append to list
                train_rmse = np.sqrt(total_loss / len(train_loader))
                training_rmse.append(train_rmse)

                # Compute RMSE for validation and append to list
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_val)
                    val_loss = self.loss_fn(val_predictions, Y_val).item()
                val_rmse = np.sqrt(val_loss)
                validation_rmse.append(val_rmse)
                
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X = self._preprocessor(x, training=False)  # Preprocess inputs
        
        self.model.eval()  # Set the model to evaluation mode
        
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.numpy()


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training=False)  # Preprocess inputs
        Y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)  # Ensure Y is the correct shape
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        
        mse = mean_squared_error(Y.numpy(), predictions.numpy())
        rmse = np.sqrt(mse)  # Compute the RMSE from MSE
        return rmse


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


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



def perform_hyperparameter_search(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print(f"\nRegressor error: {error}\n")


if __name__ == "__main__":
    example_main()

