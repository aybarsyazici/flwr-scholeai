# data_generation/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install curl and build-essential
RUN apt-get update && \
    apt-get install -y curl build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Use the official installer script, which is the most reliable method.
RUN curl -fsSL https://ollama.com/install.sh | sh

# The installer places the binary in /usr/local/bin, which is already in the PATH.

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml ./
RUN uv sync

COPY ./data_generation/prompts.py ./
COPY ./data_generation/generate_data.py ./

CMD ["uv", "run", "python", "generate_data.py"]