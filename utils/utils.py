import matplotlib.pyplot as plt


class PlotUtils:
    @staticmethod
    def plot_training_curve(train_losses, val_losses, title="Training Curve"):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_regression_results(
        inputs, true_values, predicted_values, title="Regression Results"
    ):
        plt.figure(figsize=(10, 6))
        plt.scatter(inputs.numpy(), true_values.numpy(), label="True Values", alpha=0.6)
        plt.scatter(
            inputs.numpy(),
            predicted_values.numpy(),
            label="Predicted Values",
            alpha=0.6,
        )
        # draw regression line
        sorted_indices = inputs[:, 0].argsort()
        plt.plot(
            inputs[sorted_indices].numpy(),
            predicted_values[sorted_indices].numpy(),
            color="red",
            linewidth=2,
            label="Regression Line",
        )

        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
