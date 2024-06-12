import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Smart Home Setup Assistant")
st.markdown("Welcome to our Smart Home Setup Assistant! Our AI-powered assistant is here to guide you through setting up and integrating. Just tell us your concerns or questions, and we'll provide clear and tailored instructions to help you out. Let's make your smart home experience effortless!")
input = st.text_input("Please enter your concerns:",placeholder=f"""Type here""")

open_ai_text_completion_model = OpenAIModel(
    api_key=st.secrets["apikey"],
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def generation(input):
    generator_agent = Agent(
        role="Expert SMART HOME SETUP ASSISTANT",
        prompt_persona=f"Your task is to GUIDE users through the process of SETTING UP specific devices, CREATING automation routines, and INTEGRATING these devices seamlessly.")
    prompt = f"""
You are an Expert SMART HOME SETUP ASSISTANT. Your task is to GUIDE users through the process of SETTING UP specific devices, CREATING automation routines, and INTEGRATING these devices seamlessly.

Proceed with the following steps:

1. IDENTIFY the specific smart home devices the user wants to set up, such as smart lights, thermostats, security cameras, or voice assistants.

2. PROVIDE clear and SIMPLE INSTRUCTIONS for the physical setup of each device, ensuring that you cover any initial installation requirements.

3. EXPLAIN how to CONNECT each device to the user's home Wi-Fi network or any necessary hubs.

4. DEMONSTRATE how to DOWNLOAD and use the corresponding apps for device configuration and control.

5. OUTLINE steps to CREATE basic automation routines that serve common needs like morning wake-up routines or energy-saving settings when no one is home.

6. ILLUSTRATE how to INTEGRATE multiple devices so they can work together seamlessly, such as having motion sensors trigger lights or setting up voice commands through a smart speaker.

7. EMPHASIZE the importance of SECURITY and PRIVACY settings in their smart home setup.

You MUST communicate these instructions in a way that is EASY TO UNDERSTAND and eliminates any potential frustration from the user's experience. 
"""

    generator_agent_task = Task(
        name="Generation",
        model=open_ai_text_completion_model,
        agent=generator_agent,
        instructions=prompt,
        default_input=input,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
    ).execute()

    return generator_agent_task 
   
if st.button("Assist!"):
    solution = generation(input)
    st.markdown(solution)

with st.expander("ℹ️ - About this App"):
    st.markdown("""
    This app uses Lyzr Automata Agent . For any inquiries or issues, please contact Lyzr.

    """)
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack",
                   url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                   use_container_width=True)