import os
import json
import yaml
from typing import Dict, Any

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.json'):
            return json.load(f)
        elif config_path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f)
        else:
            raise ValueError("Config file must be JSON or YAML")

config_path = os.path.join(os.path.curdir, "roleplay_chatbot/agent_config.yaml")
config = load_config(config_path)

# Initialize model
model = LiteLlm(
    model=config["model"]["name"],
    api_key=os.getenv(config["model"]["api_key_env"]),
)

# Create agent with config
root_agent = Agent(
    name="roleplay_chatbot",
    model=model,
    description=config["agent"]["description"],
    instruction=config["agent"]["instruction"],
    tools=config["agent"].get("tools", []),
)