
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.fc1 = nn.Linear(19,20)
        self.fc2 = nn.Linear(20,1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.fc1 = nn.Linear(19, 20)                  # Fully connected layer 1
        self.relu = nn.ReLU()                         # Activation function
        self.fc2 = nn.Linear(20, 1)                   # Fully connected layer 2
        

    def forward(self, x):
        # Define forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)        
        return x       


class DeepNeuralNetwork(nn.Module):
  def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepNeuralNetwork, self).__init__()
        
        # Define the architecture
        self.hidden_layers = nn.ModuleList()  # List to hold hidden layers
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))  # Input to first hidden layer
        
        # Add subsequent hidden layers
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)  # Last hidden to output layer

        # Activation function
        self.activation = nn.ReLU()  # Using ReLU for all layers
    
  def forward(self, x):
        # Forward pass through hidden layers
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        # Forward pass through the output layer
        x = self.output_layer(x)
        return x