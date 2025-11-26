import torch
from torch import nn
from dataclasses import dataclass, field

@dataclass
class MLPConfig:
    input: int = 109
    output: int = 10
    neurons: list[int] = field(default_factory=[1, 2, 3])
    layers: int = 3

class NeuralNetwork(nn.Module):
  def __init__(self, cfg: MLPConfig):
    
      super().__init__()
      self.input= cfg.input
      self.output= cfg.output
      self.layers= cfg.layers
      self.neurons=cfg.neurons
      assert len(self.neurons)==self.layers, "Número de capas y neuronas no coinciden"

      self.linnears = []

      for i, j in enumerate(self.neurons):
        if i == 0:
          self.linnears.append(nn.Linear(self.input,j))
          self.linnears.append(nn.ReLU())
        elif i <= self.layers:
          self.linnears.append(nn.Linear(self.neurons[i-1],j))
          self.linnears.append(nn.ReLU())
          
      self.linnears.append(nn.Linear(j,self.output))


  def forward(self, x):
      for layer in self.linnears:
          x = layer(x)

      return x