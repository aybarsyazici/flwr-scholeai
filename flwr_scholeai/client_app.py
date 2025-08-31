import flwr as fl
from flwr.common import Context
import torch
import json
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer
from datasets import load_from_disk
import datasets

# --- Configuration & Setup (from your last working version) ---
MODEL_NAME = os.environ.get("HF_MODEL_NAME", "meta-llama/Llama-3.2-1B")
DATASET_PATH = "/app/data/curriculum_preferences"
MAX_SEQ_LENGTH = 8192
CACHE_DIR = os.environ.get("HF_DATASETS_CACHE", "/cache")
os.makedirs(CACHE_DIR, exist_ok=True)
datasets.config.HF_DATASETS_CACHE = CACHE_DIR
datasets.config.IN_MEMORY_MAX_SIZE = 6 * 1024**3 # Set a 6GB RAM limit for in-memory datasets
print(f"Hugging Face datasets cache directory manually set to: {datasets.config.HF_DATASETS_CACHE}")

# --- The ML Logic: A Flower NumPyClient ---
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, tokenizer, dataset_shard, device):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_shard = dataset_shard
        self.device = device
        self.model.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("Client: Starting local training (fit).")
        self.set_parameters(parameters)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        training_args = RewardConfig(
            output_dir=os.path.join(CACHE_DIR, "results_reward"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            logging_steps=1,
            learning_rate=5e-6,
            save_strategy="no",
            report_to="none",
            fp16=(device.type == 'cuda'),
            use_cpu=(device.type == 'cpu'),
            max_length=MAX_SEQ_LENGTH,
        )
        
        trainer = RewardTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset_shard,
            processing_class=self.tokenizer,
        )
        
        print("Client: RewardTrainer training starting...")
        trainer.train()
        print("Client: RewardTrainer training finished.")
        return self.get_parameters({}), len(self.dataset_shard), {}

    def evaluate(self, parameters, config):
        # We can add a proper evaluation later if needed
        return 0.0, len(self.dataset_shard), {"accuracy": 0.0}

# --- The App Definition ---
def client_fn(context: Context):
    """This function is executed by the ClientApp to create a FlowerClient."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Client device: {device}")
    
    # Load data partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    print(f"Loading data from path: {DATASET_PATH}")
    dataset = load_from_disk(DATASET_PATH)["train"]
    dataset_shard = dataset.shard(num_shards=num_partitions, index=partition_id, contiguous=True)
    
    def format_for_reward_trainer(example):
        prompt = example["prompt"]
        chosen_str = json.dumps(example["chosen"], indent=2)
        rejected_str = json.dumps(example["rejected"], indent=2)
        return {"chosen": prompt + "\n\n---\n\n" + chosen_str, "rejected": prompt + "\n\n---\n\n" + rejected_str}

    print("Formatting data for reward trainer...")
    dataset_shard = dataset_shard.map(format_for_reward_trainer, remove_columns=["prompt"])
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Return a FlowerClient instance
    print("Returning FlowerClient")
    return FlowerClient(model, tokenizer, dataset_shard, device).to_client()

app = fl.client.ClientApp(client_fn)