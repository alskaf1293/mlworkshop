from manim import *
import torch
import torch.nn as nn
import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

np.random.seed(0)

neurons = np.array([0,1,1,1,0,0,1,1,0,0,1,1,1])

weights = np.random.uniform(-1,1, len(neurons))

class ValueIncreasingScene(Scene):
    def construct(self):
        # Create a ValueTracker and set its initial value
        value_tracker = ValueTracker(0)
        neuron = Circle(radius=0.5, fill_color=BLACK, stroke_color=WHITE, fill_opacity=1)
        # Function to create a text object displaying the current value
        def value_label():
            return Text(f"{value_tracker.get_value():.2f}", font_size=36, fill_color=WHITE).next_to(neuron, UP)

        # Create initial value text
        value_text = value_label()

        self.add(neuron)
        self.add(value_text)

        # Function to update the text as the value changes
        def update_value_text(mob):
            new_value_text = value_label()
            mob.become(new_value_text)

        # Add updater to the value text
        value_text.add_updater(update_value_text)

        # Animate the change in value
        self.play(value_tracker.animate.set_value(1), neuron.animate.set_fill(WHITE), value_text.animate.set_fill(BLACK), run_time=1)

        self.wait(1)

        self.play(value_tracker.animate.set_value(0), neuron.animate.set_fill(BLACK), value_text.animate.set_fill(WHITE), run_time=1)

        self.play(Unwrite(value_text), run_time=1.5)

        self.wait(1)
        # Remove updater after animation
        value_text.remove_updater(update_value_text)


        ##### CIRCLES
        circles_group = VGroup()

        # Create circles based on random values and add them to the VGroup
        for value in neurons:
            color = interpolate_color(BLACK, WHITE, value)
            circle = Circle(radius=0.1, fill_color=color, fill_opacity=1, stroke_color=WHITE, stroke_width=0.7)
            circles_group.add(circle)

        # Arrange the circles vertically (downwards)
        circles_group.arrange(DOWN, buff=0.1)  # `buff` is the buffer space between circles
        circles_group.shift(1.5*LEFT)

        ##### LINES
        final_value = sigmoid(np.dot(weights, neurons))
        final_color = interpolate_color(BLACK, WHITE, final_value)

        perceptron_neuron = Circle(radius=0.1, fill_color=final_color, fill_opacity=1, stroke_color=WHITE).shift(1.5*RIGHT)

        lines_group = VGroup()

        # Create lines based on weights
        for i, (circle, weight) in enumerate(zip(circles_group, weights)):
            line = Line(circle.get_center(), perceptron_neuron.get_center(), stroke_width=2)
            line_color = BLUE if weight > 0 else RED
            line.set_color(line_color)
            lines_group.add(line)


        self.play(Write(circles_group), Transform(neuron,circles_group))

        self.play(Write(perceptron_neuron))

        # Show the circles and lines together
        self.play(Write(lines_group))

        self.wait(1)

        nnvgroup = VGroup()

        nnvgroup.add(lines_group)
        nnvgroup.add(circles_group)
        nnvgroup.add(perceptron_neuron)

        self.play(nnvgroup.animate.to_edge(LEFT).shift(RIGHT))

