import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import pickle

def preprocess_images(image_dir,):
    X = []
    y = []
    
    print("Beginning to parse image files...")

    for filename in os.listdir(image_dir):
        if filename.endswith(".png"): 
            img_path = os.path.join(image_dir, filename)
            
            img = Image.open(img_path)
            img_array = np.array(img).flatten()
            
            X.append(img_array)
            
            percentage = float(filename.split('.')[2])
            y.append(percentage)

    print("Image file parsing complete. Starting to train the model.")

    return np.array(X), np.array(y)

def train(pca_dimension, hidden_layer_sizes=(256,128,64), show_plot=False, verbose=False):
    image_directory = "dataset"  
    X, y = preprocess_images(image_directory)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_reduced = X_scaled

    if pca_dimension > 0:
        pca = PCA(n_components=pca_dimension)
        X_reduced = pca.fit_transform(X_scaled)

    X_train, X_temp, y_train, y_temp = train_test_split(X_reduced, y, test_size=0.5, random_state=1)
    X_val, X_test, y_val, y_test  = train_test_split(X_temp, y_temp, test_size=0.20, random_state=1)

    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,  
                    activation='relu',  
                    solver='adam',  
                    learning_rate='adaptive',  
                    max_iter=500,  
                    random_state=42, verbose=verbose)

    mlp.fit(X_train, y_train)

    y_pred_train = mlp.predict(X_train)
    y_pred_val = mlp.predict(X_val)
    y_pred_test = mlp.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_val = mean_squared_error(y_val, y_pred_val)
    mse_test = mean_squared_error(y_test, y_pred_test)

    mae_val = mean_absolute_error(y_val, y_pred_val)

    # Saving the model
    with open(f"models/{pca_dimension}-{len(hidden_layer_sizes)}-{mse_val.round(2)}-{mae_val.round(2)}.pkl", "wb") as f:
        pickle.dump(pca, f)

    if verbose:
        print(f"Mean Squared Error (train): {mse_train}")
        print(f"Mean Squared Error (validate): {mse_val}")
        print(f"Mean Squared Error (test): {mse_test}")
        print(f"Mean Absolute Error: {mae_val}")
            

    if show_plot:
        figure, axis = plt.subplots(2, 2)

        # Predicted vs Actual Values
        axis[0, 0].scatter(y_val, y_pred_val, color='blue', alpha=0.6)
        axis[0, 0].plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
        axis[0, 0].set_title('Predicted vs Actual Values')
        axis[0, 0].set_xlabel('Actual Percentage')
        axis[0, 0].set_ylabel('Predicted Percentage')
        axis[0, 0].grid(True)

        # Residuals (Errors) Plot
        residuals = y_val - y_pred_val
        axis[1, 0].scatter(y_pred_val, residuals, color='purple', alpha=0.6)
        axis[1, 0].axhline(y=0, color='r', linestyle='--')
        axis[1, 0].set_title('Residuals Plot')
        axis[1, 0].set_xlabel('Predicted Percentage')
        axis[1, 0].set_ylabel('Residuals (Actual - Predicted)')
        axis[1, 0].grid(True)

        # Histogram of Errors (Residuals)
        axis[1, 1].hist(residuals, bins=20, color='green', edgecolor='black', alpha=0.7)
        axis[1, 1].set_title('Distribution of Residuals')
        axis[1, 1].set_xlabel('Residual (Error)')
        axis[1, 1].set_ylabel('Frequency')
        axis[1, 1].grid(True)

        # MSE of train, validation and test sets
        mse_values = [mse_train, mse_val, mse_test]
        labels = ['Train MSE', 'Validation MSE', 'Test MSE']
        axis[0, 1].bar(labels, mse_values, color=['blue', 'orange', 'green'])
        axis[0, 1].set_title('MSE for Train, Validation, and Test Sets')
        axis[0, 1].set_xlabel('Dataset')
        axis[0, 1].set_ylabel('Mean Squared Error')

        plt.show()


if __name__ == '__main__':
    train(-1, (512, 256, 128, 64), True, True)