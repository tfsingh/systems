import ray
import ray.dag
import time
import torch
import torch.nn as nn
import torch.optim as optim

@ray.remote
class RayMLPLayerOne:
    def __init__(self, input_size, hidden_size):
        self.model = self.SimpleMLP(input_size, hidden_size)

    class SimpleMLP(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(RayMLPLayerOne.SimpleMLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            return x

    def forward(self, input_tensor):
        return self.model(input_tensor)

@ray.remote
class RayMLPLayerTwo:
    def __init__(self, hidden_size, output_size):
        self.model = self.SimpleMLP(hidden_size, output_size)

    class SimpleMLP(nn.Module):
        def __init__(self, hidden_size, output_size):
            super(RayMLPLayerTwo.SimpleMLP, self).__init__()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            return self.fc2(x)

    def forward(self, input_tensor):
        return self.model(input_tensor)

input_size = 10
hidden_size = 20
output_size = 1

start = time.perf_counter()
layer_one = RayMLPLayerOne.remote(input_size, hidden_size)
layer_two = RayMLPLayerTwo.remote(hidden_size, output_size)
for _ in range(10):
    example_input = torch.randn(5, input_size)
    intermediate_output = ray.get(layer_one.forward.remote(example_input))
    output = ray.get(layer_two.forward.remote(intermediate_output))
print("dynamic took", time.perf_counter() - start)


compiled = start = None
with ray.dag.InputNode() as inp:
    intermediate = layer_one.forward.bind(inp)
    dag = layer_two.forward.bind(intermediate)
    compiled = dag.experimental_compile()

    start = time.perf_counter()
    for _ in range(10):
        example_input = torch.randn(5, input_size)
        output = ray.get(compiled.execute(example_input))
print("compiled took", time.perf_counter() - start)

compiled.teardown()



