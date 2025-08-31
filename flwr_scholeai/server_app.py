import flwr as fl
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
import torch
import argparse
import os
from transformers import AutoModelForSequenceClassification

# We load the model here just once to get its initial shape for the strategy
MODEL_NAME = os.environ.get("HF_MODEL_NAME", "meta-llama/Llama-3.2-1B")

def get_initial_parameters():
    """Load model and return initial parameters"""
    print(f"Server: Loading initial parameters from {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, trust_remote_code=True
    )
    # Return the parameters as a list of NumPy arrays
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

# Define a simple weighted averaging function for metrics
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context) -> ServerAppComponents:
    """This function is executed by the ServerApp to get the components."""
    # The run_config is sent by the `flwr run` command
    num_rounds = context.run_config["num-server-rounds"]
    min_fit_clients = context.run_config["min-fit-clients"]
    min_available_clients = context.run_config["min-available-clients"]

    # Define the strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        initial_parameters=ndarrays_to_parameters(get_initial_parameters()),
        evaluate_metrics_aggregation_fn=weighted_average, # Add a simple metric aggregation
    )

    # Configure the server
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create the ServerApp
app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    print("This script is designed to be run by 'flower-serverapp', not directly.")