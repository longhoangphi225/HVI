from torch import nn
import torch.nn.functional as F
import torch
class Hyper_mlp(nn.Module):
      "Hypernetwork"

      def __init__(self, ray_hidden_dim=100, out_dim = 2,n_tasks=2,num_hidden_layer=2,last_activation='relu'):
            super().__init__()
            self.out_dim = out_dim
            self.n_tasks = n_tasks
            self.ray_hidden_dim = ray_hidden_dim
            self.num_hidden_layer = num_hidden_layer
            self.last_activation = last_activation
            self.input_layer =  nn.Sequential(nn.Linear(self.n_tasks, self.ray_hidden_dim),nn.ReLU(inplace=True))
            self.hidden_layer = nn.ModuleList([nn.Linear(self.ray_hidden_dim, self.ray_hidden_dim) for i in range(self.num_hidden_layer)])
            self.output_layer =  nn.Linear(self.ray_hidden_dim, self.out_dim)
      def forward(self, ray):
            x = self.input_layer(ray)
            for i in range(self.num_hidden_layer):
                  x = self.hidden_layer[i](x)
                  x = F.relu(x)
            x = self.output_layer(x)
            if self.last_activation == 'relu':
                  x = F.relu(x)
            elif self.last_activation == 'sigmoid':
                  x = F.sigmoid(x)
            elif self.last_activation == 'softmax':
                  x = F.softmax(x)
            else:
                  x = x
            x = x.unsqueeze(0)
            return x
