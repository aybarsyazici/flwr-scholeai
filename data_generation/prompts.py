# data_generation/prompts.py

def get_generation_prompt(topic: str, constraints: str) -> str:
    """
    Creates a prompt for an LLM to generate a good curriculum and a bad curriculum
    for a given topic and set of constraints.
    """
    return f"""
You are an expert instructional designer and a data generation specialist.
Your task is to create a JSON object containing a 'chosen' curriculum and a 'rejected' curriculum based on a user's request.

**User Request:**
- Topic: "{topic}"
- Constraints: "{constraints}"

**Your Task:**
Generate two complete curriculum JSON objects.

1.  **'chosen' curriculum:** This curriculum MUST be excellent. It should perfectly follow the user's request, be logically structured, and adhere to all constraints.
2.  **'rejected' curriculum:** This curriculum MUST be flawed in a clear and obvious way. Introduce one of the following flaws:
    *   **Topic Mismatch:** Make the curriculum about a related but incorrect topic (e.g., if the user asks for "Kubernetes", generate a curriculum for "Docker Swarm").
    *   **Constraint Violation:** Ignore one of the user's key constraints (e.g., if they ask for "video-based learning", make all modalities "text").
    *   **Poor Structure:** Make the curriculum illogical or poorly organized, i.e. advanced topics for a beginner.
    *   **Incorrect Format:** Generate a curriculum that is missing required JSON fields or has incorrect data types.

The final output MUST be a single JSON object containing two keys: "chosen" and "rejected". Each key should hold a valid curriculum JSON object. Do not add any text or explanations before or after the main JSON object.

**Example Curriculum JSON Format:**
{{
  "title": "...",
  "description": "...",
  "company": "...",
  "time_limit": 90,
  "sections": [
    {{
      "title": "...",
      "order": 0,
      "description": "...",
      "subsections": [
        {{
          "title": "...",
          "order": 0,
          "modality": "interactive",
          "estimated_time_min": 15,
          "description": "..."
        }}
      ]
    }}
  ]
}}
"""