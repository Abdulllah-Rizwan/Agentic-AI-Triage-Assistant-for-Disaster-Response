from google.adk.agents import Agent
from utils import get_prompts


MODEL = 'gemini-2.5-flash'

triage_agent = Agent(
    name = 'Triage_classifier',
    model = MODEL,
    description = 'You are a triage classifier. You will classify the symptoms of the patient into a triage level and send back the results to the guidance agent.',
    instruction = get_prompts(agent='triage_classifier'),
)