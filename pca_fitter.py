import model

def main():
    print("Test different component counts for PCA to see which outputs the best training results using MLP.")
    for i in range(25, 670, 25):
        hidden_layer_sizes = (512, 256, 128, 64)
        print(f"Training with: PCA components: {i}, Hidden layer sizes: {hidden_layer_sizes}")
        model.train(i, (512, 256, 128, 64), False, False)

if __name__ == '__main__':
    main()