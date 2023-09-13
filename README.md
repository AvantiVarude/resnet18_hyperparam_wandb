# README: Training a ResNet18 Model for Classification on Even Classes of CIFAR-10 using Weights and Biases (Wandb)

In this project, we will train a ResNet18 deep learning model for image classification on the even classes of the CIFAR-10 dataset using the Weights and Biases (Wandb) framework. We will experiment with different hyperparameters to optimize the model's performance.

## Prerequisites

Before you begin, make sure you have the following prerequisites installed:

- Python 3.x
- PyTorch
- torchvision
- Wandb (Weights and Biases)

You can install these dependencies using pip:

```bash
pip install torch torchvision wandb
```

## Getting Started

1. Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/cifar10-resnet18-wandb.git
cd cifar10-resnet18-wandb
```

2. Set up your Wandb account and login:

   - If you don't have a Wandb account, sign up at [https://wandb.ai/](https://wandb.ai/).
   - Install Wandb by running `pip install wandb`.
   - Run `wandb login` and follow the instructions to authenticate your account.

3. Prepare the CIFAR-10 dataset:

   You can use the built-in torchvision datasets to download and prepare the CIFAR-10 dataset. Refer to the official PyTorch documentation for guidance on how to load the dataset: [https://pytorch.org/vision/stable/datasets.html#cifar](https://pytorch.org/vision/stable/datasets.html#cifar)

4. Configure Experiment Hyperparameters:

   Open the `config.yaml` file and adjust the hyperparameters as needed. You can customize parameters such as learning rate, batch size, number of epochs, etc. Here is an example configuration:

   ```yaml
   experiment_name: resnet18_cifar10_even_classes
   num_epochs: 50
   batch_size: 128
   learning_rate: 0.001
   weight_decay: 0.0001
   ```
   
   Modify and add more hyperparameters as needed for your experiments.

5. Training:

   Run the training script, specifying the path to your configuration file:

   ```bash
   python train.py --config config.yaml
   ```

   This script will train the ResNet18 model on the even classes of CIFAR-10 using the hyperparameters defined in `config.yaml`. Wandb will automatically log metrics, losses, and other important information during training.

6. Monitoring and Visualizing Results:

   You can monitor and visualize the training progress on the Wandb dashboard. Log in to your Wandb account, navigate to the project, and you will find all the logged metrics and visualizations for your experiments.

7. Experimentation:

   To experiment with different hyperparameters, simply modify the `config.yaml` file and re-run the training script. Wandb will keep track of all your experiments and their results for easy comparison.

## Conclusion

This README provides a step-by-step guide on how to train a ResNet18 model for image classification on the even classes of CIFAR-10 using Wandb to log and visualize different hyperparameter experiments. You can extend this project by exploring more advanced hyperparameter tuning techniques and model architectures to achieve better classification results.
