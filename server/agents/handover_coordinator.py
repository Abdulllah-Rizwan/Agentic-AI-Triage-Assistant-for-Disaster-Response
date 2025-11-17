from google.adk.agents import Agent
from utils import get_prompts

MODEL = 'gemini-2.5-flash'

handover_coordinator = Agent(
    name='handover_coordinator',
    model=MODEL,
    description='You are a handover coordinator. You will generate the SOAP report and send it to the Doctor asking him to onboard and communicate with the patient directly.',
    instruction=get_prompts(agent='handover_coordinator'),
    tools = [ 
        #send email wala tool idhar aega. Will implement later
        ]

)