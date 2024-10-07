import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os


def get_sorted_results(PATH):
    with os.scandir(PATH) as models:
        models_parsed = [[float(y) for  y in x.name[:-4].split('-')] for x in models]
        models_sorted = sorted(models_parsed, key=lambda model : model[0])
    return models_sorted    

def read_model_results(PATH):
    mse = []
    mae = []
    component_count = []
    for file in get_sorted_results(PATH):
            model_data = [float(x) for x in file]
            component_count.append(model_data[0])
            mse.append(model_data[2])
            mae.append(model_data[3])

    return (mse, mae, component_count)


def main():
    PATH = "models/"
    mse, mae, component_count = read_model_results(PATH)

    xpoints = np.array(component_count)
    ypoints = np.array(mse)

    c = np.polyfit(xpoints, ypoints, 1)
    linear_fit = np.poly1d(c)

    plt.plot(xpoints, ypoints, xpoints, linear_fit(xpoints), '--k')

    plt.title("Changes in MSE due to variation in PCA component count")

    plt.xlabel("Number of PCA components")
    plt.ylabel("Mean squared error of MLPRegressor")

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=ax.get_xticks().size * 2))

    plt.show()

if __name__ == '__main__':
    main()