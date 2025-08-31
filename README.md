# Federated Reward Model Training for Curriculum Quality

This project demonstrates how to perform federated reward model training using Flower's modern `App` architecture and the Hugging Face TRL library. The goal is to train a model that can distinguish between high-quality ("chosen") and low-quality ("rejected") AI-generated learning curriculums.

The entire federated learning infrastructure is orchestrated using Docker Compose, modeling a realistic deployment scenario with built-in support for advanced privacy-preserving techniques like Secure Aggregation and Differential Privacy.

## Project Structure

The project follows the structure recommended by the official Flower examples:

```
.
├── compose.yml              # Main Docker Compose file for infrastructure
├── data/                      # Holds the generated dataset (created automatically)
├── data_generation/         # Scripts and config for synthetic data generation
│   ├── generate_data.py
│   ├── prompts.py
│   └── pyproject.toml
├── flwr_scholeai/           # The main Flower App Python package
│   ├── __init__.py
│   ├── client_app.py      # Defines the ClientApp and ML logic
│   └── server_app.py      # Defines the ServerApp and FL strategy
├── pyproject.toml           # Defines project dependencies and Flower App config
└── README.md                # This file
```

## Prerequisites

Before you begin, ensure you have the following installed on your local machine:

1.  **Docker & Docker Compose v2:** [Install Docker](https://docs.docker.com/engine/install/)
2.  **Python 3.10+** and `pip`.
3.  **Flower CLI:** `pip install flwr`
4.  **Ollama (Optional):** For generating data locally. [Install Ollama](https://ollama.com/).
5.  **OpenAI API Key (Optional):** For generating data quickly via the API.

## Setup & Execution

The process is divided into two main stages: one-time data generation and repeatable federated training.

### Step 1: Initial Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd flwr-scholeai
    ```

2.  **Create an Environment File:**
    Create a `.env` file in the project root for your secrets and configuration by copying the example.
    ```bash
    cp .env.example .env
    ```
    Then, edit the `.env` file and add your actual API keys (`OPENAI_API_KEY`, `HF_TOKEN`).

### Step 2: Generate the Training Dataset

This step only needs to be run once. Choose one of the following options.

#### Option A: Fast Generation with OpenAI
1.  In your `.env` file, ensure `GENERATION_BACKEND=openai`.
2.  Run the command:
    ```bash
    docker-compose --profile generate-openai up --build
    ```

#### Option B: Local Generation with Ollama (Slower)
1.  Make sure your local Ollama application is running.
2.  In your `.env` file, ensure `GENERATION_BACKEND=ollama`.
3.  Run the command:
    ```bash
    docker-compose --profile generate-ollama up --build
    ```

### Step 3: Run the Federated Training

Once the data exists in the `./data` directory, you can start the federated training.

1.  **Export the Project Directory:**
    This variable tells Docker Compose where to find your `pyproject.toml` to build the app images.
    ```bash
    export PROJECT_DIR=$(pwd)/flwr_scholeai
    ```

2.  **Start the Flower Infrastructure:**
    This command starts the `SuperLink` and `SuperNode`s in the background.
    ```bash
    docker compose up --build -d
    ```

3.  **Launch the Federated Run:**
    This command uses the Flower CLI to submit your `ServerApp` and `ClientApp` to the running infrastructure and starts the experiment.
    ```bash
    flwr run . local-deployment --stream
    ```
    You will see the logs from the server and clients stream to your terminal as they perform federated training.

## Understanding the Architecture

This project uses Flower's modern deployment architecture, which decouples the long-running **Infrastructure** from the short-lived **Application Logic**.

![Flower Architecture](https://flower.ai/docs/framework/_images/flower-architecture-basic-architecture.svg)


*   **Infrastructure Components (Started by `docker compose up`):**
    *   `SuperLink`: The central message broker and task dispatcher.
    *   `SuperNode`: A persistent agent running on a client machine, waiting for tasks.
*   **Application Logic Components (Referenced by `flwr run`):**
    *   `ServerApp` (in `server_app.py`): Contains your `FedAvg` strategy and server-side configuration.
    *   `ClientApp` (in `client_app.py`): Contains your ML training code (`RewardTrainer`).

The `flwr run` command acts as the orchestrator, telling the `SuperLink` to start a new federated run using your defined `ServerApp` and `ClientApp`.

![image](https://flower.ai/docs/framework/_static/flower-network-diagram-subprocess.svg)

## Privacy and Security Features

The architecture is designed to be fully compatible with advanced privacy-preserving protocols, as required by modern federated learning systems.

### Secure Communication with TLS

Encrypted communication between all Flower components can be enabled through configuration. The implementation plan is as follows:
1.  **Generate Certificates:** Use the `certs.yml` utility from Flower's examples to generate self-signed certificates.
2.  **Update Docker Compose:** Use a `with-tls.yml` override file to mount certificates and remove the `--insecure` flags from the `SuperLink` and `SuperNode` services.
3.  **Update `pyproject.toml`:** The federation definition for `flwr run` would be updated to include the path to the root certificate.

### Secure Aggregation (SecAgg+)

To prevent the central server from inspecting individual model updates, Flower's `SecAgg+` protocol is implemented. This ensures that the server can only see the combined, aggregated result, not the contribution of any single client.

*   **Server-Side (`server_app.py`):** The `FedAvg` strategy is wrapped with the `flwr.server.strategy.secagg.SecAgg` wrapper.
*   **Client-Side (`client_app.py`):** The `ClientApp` is initialized with the `flwr.client.mod.secaggplus_mod`, enabling it to participate in the secure aggregation protocol.

### Differential Privacy (DP)

To provide formal privacy guarantees against data re-identification, Central Differential Privacy is implemented on the server. This is achieved by adding noise to the aggregated updates before they are applied to the global model.

*   **Server-Side (`server_app.py`):** The strategy (already wrapped with `SecAgg`) is further wrapped with `flwr.server.strategy.dp.DifferentialPrivacyServerSideFixedClipping`. This component performs L2 norm clipping on incoming updates and adds calibrated Gaussian noise to the final sum, controlled by a `noise_multiplier` parameter.

## Cleanup

To stop all running services and remove the containers, run:
```bash
docker compose down
```
To also remove the Docker volumes (like the client caches), add the `-v` flag:
```bash
docker compose down -v
```
```