from manim import *
import torch
import torch.nn as nn
import numpy as np

# Dummy PyTorch model for illustration
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(20, 10)
        self.layer2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 4)


    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return torch.softmax(self.output(x))

def extract_model_layers(model):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):  # Check if module is an instance of nn.Linear
            layers.append(module)
    return layers

# Manim Scene that visualizes the PyTorch model
class ModelVisualization(Scene):
    def construct(self):
        model = SimpleNet()
        layers = extract_model_layers(model)

        # Create a vertical list of layers to display
        layer_visuals = VGroup()
        all_neurons = []  # To store references to all neurons in each layer

        for layer in layers:
            layer_neurons = VGroup()
            neurons = []  # To store neurons of the current layer
            for x in range(layer.out_features):
                neuron = Circle(radius=0.05, color=BLUE_B, fill_opacity=1)
                layer_neurons.add(neuron)
                neurons.append(neuron)  # Store reference to the neuron

            layer_neurons.arrange(RIGHT, buff=0.1)
            layer_visuals.add(layer_neurons)
            all_neurons.append(neurons)  # Store neurons of this layer

        # Arrange layers vertically with some space
        layer_visuals.arrange(DOWN, center=True, buff=0.5)

        # Animate the addition of each layer to the visualization
        for layer_visual in layer_visuals:
            self.play(Write(layer_visual))
            self.wait(0.5)

        # Draw lines between neurons of adjacent layers
        for i in range(1, len(all_neurons)):
            connections = VGroup()  # Group to hold all the lines/connections for this layer
            for neuron in all_neurons[i]:
                for prev_neuron in all_neurons[i - 1]:
                    line = Line(neuron.get_center(), prev_neuron.get_center()).set_stroke(width=1)
                    connections.add(line)

            self.play(Create(connections))  # Create all lines at once

# To run this scene, use the following command in your terminal:
# manim -pql script_name.py ModelVisualization
