from handover_coordinator import handover_coordinator
from triage_agent import triage_agent
from google.adk.agents import Agent
from utils import get_prompts


MODEL = 'gemini-2.5-flash'

guidance_agent = Agent(
    name='guidance_agent',
    model=MODEL,
    description='You are a guidance agent. You will provide guidance to the user based on the symptoms collected by symptom agent. Use triage_Agent tool to get the urgency of the case and only then use handover_coordinator tool to generate the SOAP report and handover the patient to the doctor.',
    instruction=get_prompts(agent_name='guidance_agent'),
    tools=[handover_coordinator,triage_agent]
)