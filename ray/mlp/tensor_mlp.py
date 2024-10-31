import torch
import torch.nn as nn
import ray

@ray.remote
class TensorParallelLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        features = out_features // 2
        self.weight = nn.Parameter(torch.randn(in_features, features))
        self.bias = nn.Parameter(torch.randn(features))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias

@ray.remote
class TensorParallelMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.fc1_part1 = TensorParallelLinear.remote(input_size, hidden_size)
        self.fc1_part2 = TensorParallelLinear.remote(input_size, hidden_size)

        self.fc2_part1 = TensorParallelLinear.remote(hidden_size, output_size)
        self.fc2_part2 = TensorParallelLinear.remote(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        fc1_out_part1 = self.fc1_part1.forward.remote(x)
        fc1_out_part2 = self.fc1_part2.forward.remote(x)

        fc1_out = torch.cat(ray.get([fc1_out_part1, fc1_out_part2]), dim=1)
        x = self.relu(fc1_out)

        fc2_out_part1 = self.fc2_part1.forward.remote(x)
        fc2_out_part2 = self.fc2_part2.forward.remote(x)

        output = torch.cat(ray.get([fc2_out_part1, fc2_out_part2]), dim=1)
        return output

def run_tensor_parallel_mlp():
    ray.init(ignore_reinit_error=True)

    input_size = 10
    hidden_size = 20
    output_size = 2

    distributed_mlp = TensorParallelMLP.remote(input_size, hidden_size, output_size)
    example_input = torch.randn(5, input_size)

    output = ray.get(distributed_mlp.forward.remote(example_input))

    print(output)

run_tensor_parallel_mlp()
