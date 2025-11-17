from google.adk.agents import Agent
from utils import get_prompts
from guideance_agent import guidance_agent

MODEL = 'gemini-2.5-flash'

symptom_agent = Agent(
    name='symptom_collector',
    model=MODEL,
    description='You are a symptom collector agent. You will be extremely calm, kind and collect the symptoms of the patient and give them to the guidance agent so that it can give the correct guidance.',
    instruction=get_prompts('symptom_collector'),
    subagents = [guidance_agent]
)