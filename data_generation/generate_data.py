import json
import httpx
import os
import random
import time
from datasets import Dataset, DatasetDict
from openai import OpenAI
from tqdm import tqdm
from prompts import get_generation_prompt

# --- Configuration ---
# Determines which backend to use: 'ollama' or 'openai'
GENERATION_BACKEND = os.getenv("GENERATION_BACKEND", "ollama")

# Ollama settings
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("GENERATION_MODEL_NAME", "phi3:mini")

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-5-nano")

NUM_TRAIN_SAMPLES = 50
NUM_EVAL_SAMPLES = 10
OUTPUT_DATA_DIR = "./data/curriculum_preferences"

# --- Define Topics and Constraints for Generation ---
TOPICS = [
    "Introduction to Python for Data Science",
    "Advanced React Hooks",
    "Kubernetes for Beginners",
    "Machine Learning with Scikit-Learn",
    "CI/CD with GitHub Actions",
    "Docker and Containerization Fundamentals",
    "SQL for Data Analysts",
    "Building REST APIs with FastAPI",
]

CONSTRAINTS = [
    "The total time should be under 90 minutes.",
    "Focus on hands-on, interactive exercises.",
    "All learning modalities must be video-based.",
    "The target audience is experienced software engineers.",
    "The curriculum should be suitable for absolute beginners with no prior knowledge.",
    "Include a section on ethical considerations.",
]

def generate_with_ollama(prompt: str) -> dict | None:
    """Generates a data point using the Ollama backend."""
    payload = { "model": OLLAMA_MODEL, "prompt": prompt, "format": "json", "stream": False, "options": {"num_ctx": 4096} }
    try:
        with httpx.Client(timeout=600.0) as client: # 10 minute timeout
            response = client.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        response_text = response.json().get('response', '{}').strip()
        start, end = response_text.find('{'), response_text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(response_text[start:end+1])
        return None
    except Exception as e:
        print(f"[Ollama Error] {e}")
        return None

def generate_with_openai(prompt: str) -> dict | None:
    """Generates a data point using the OpenAI backend."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"[OpenAI Error] {e}")
        return None

def generate_preference_data_point(max_retries=3):
    """Orchestrator function that calls the selected backend."""
    for attempt in range(max_retries):
        topic = random.choice(TOPICS)
        constraint = random.choice(CONSTRAINTS)
        prompt = get_generation_prompt(topic, constraint)
        print(f"\nGenerating for Topic: '{topic}' using '{GENERATION_BACKEND}' (Attempt {attempt + 1}/{max_retries})")

        if GENERATION_BACKEND == 'openai':
            generated_json = generate_with_openai(prompt)
        else: # Default to ollama
            generated_json = generate_with_ollama(prompt)
        
        if generated_json and "chosen" in generated_json and "rejected" in generated_json:
            generated_json["prompt"] = f"Topic: {topic}\nConstraints: {constraint}"
            return generated_json
        
        print("Warning: Malformed or missing JSON. Retrying...")
        time.sleep(2)
        
    print(f"Failed to generate data point after {max_retries} attempts.")
    return None

def main():
    print(f"--- Starting Synthetic Dataset Generation of {NUM_TRAIN_SAMPLES} train sample(s) using backend: {GENERATION_BACKEND}")
    if GENERATION_BACKEND == 'openai' and not OPENAI_API_KEY:
        print("\nERROR: GENERATION_BACKEND is 'openai' but OPENAI_API_KEY is not set.")
        print("Please create a .env file with your key or export it.")
        return
    
    train_data = []
    # Use a tqdm progress bar with a while loop
    with tqdm(total=NUM_TRAIN_SAMPLES, desc="Generating training samples") as pbar:
        while len(train_data) < NUM_TRAIN_SAMPLES:
            data_point = generate_preference_data_point()
            if data_point:
                train_data.append(data_point)
                pbar.update(1) # Manually update the progress bar on success

    eval_data = []
    with tqdm(total=NUM_EVAL_SAMPLES, desc="Generating evaluation samples") as pbar:
        while len(eval_data) < NUM_EVAL_SAMPLES:
            data_point = generate_preference_data_point()
            if data_point:
                eval_data.append(data_point)
                pbar.update(1)

    if not train_data or not eval_data:
        print("\nFailed to generate sufficient data. Exiting.")
        return

    # Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "eval": eval_dataset
    })

    print(f"\nSuccessfully generated {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples.")
    print(f"Saving dataset to {OUTPUT_DATA_DIR}")
    dataset_dict.save_to_disk(OUTPUT_DATA_DIR)
    print("--- Dataset Generation Finished ---")

if __name__ == "__main__":
    main()