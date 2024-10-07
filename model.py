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

    pca = PCA(n_components=pca_dimension)
    X_reduced = pca.fit_transform(X_scaled)

    X_train, X_val, y_train, y_val = train_test_split(X_reduced, y, train_size=0.5, test_size=0.4, random_state=42)

    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,  
                    activation='relu',  
                    solver='adam',  
                    learning_rate='adaptive',  
                    max_iter=500,  
                    random_state=42, verbose=verbose)

    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(pca.fit_transform(X_val))

    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    # Saving the model
    with open(f"models/{pca_dimension}-{len(hidden_layer_sizes)}-{mse.round(2)}-{mae.round(2)}.pkl", "wb") as f:
        pickle.dump(pca, f)

    if verbose:
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
            

    if show_plot:
        figure, axis = plt.subplots(2, 2)

        # Predicted vs Actual Values
        axis[0, 0].scatter(y_val, y_pred, color='blue', alpha=0.6)
        axis[0, 0].plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
        axis[0, 0].set_title('Predicted vs Actual Values')
        axis[0, 0].set_xlabel('Actual Percentage')
        axis[0, 0].set_ylabel('Predicted Percentage')
        axis[0, 0].grid(True)

        # Residuals (Errors) Plot
        residuals = y_val - y_pred
        axis[1, 0].scatter(y_pred, residuals, color='purple', alpha=0.6)
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

        plt.show()