from pathlib import Path
import yaml


PROMPTS_PATH = Path(__file__).parent / 'prompts' / 'prompts.yaml'

with open(PROMPTS_PATH,'r') as file:
    PROMPTS = yaml.safe_load(file)

def get_prompts(agent_name:str) -> str:
    return PROMPTS[agent_name]['system']
