import ray
import time
import torch
import torch.nn as nn
import torch.optim as optim

@ray.remote
class RayMLPWrapper:
    def __init__(self, input_size, hidden_size, output_size):
        self.model = self.SimpleMLP(input_size, hidden_size, output_size)

    class SimpleMLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RayMLPWrapper.SimpleMLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    def forward(self, input_tensor):
        return self.model(input_tensor)

input_size = 10
hidden_size = 20
output_size = 1
mlp = RayMLPWrapper.remote(input_size, hidden_size, output_size)

start = time.perf_counter()
for _ in range(10):
    example_input = torch.randn(5, input_size)
    output = ray.get(mlp.forward.remote(example_input))
print("took", time.perf_counter() - start)



