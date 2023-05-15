import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.schema import SystemMessage
from dotenv import load_dotenv, find_dotenv
from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType

# Load environment variables
_ = load_dotenv(find_dotenv())
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

# App framework
st.title('🦜🔗 Investment Strategies Generator')
prompt = st.text_input('Enter your prompt here')

# Prompt templates
strategy_template = PromptTemplate(
    input_variables=['topic'],
    template='Write me detailed investment strategies about {topic}'
)

# Memory
strategy_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

# LLM
llm = OpenAI(temperature=0.9)
strategy_chain = LLMChain(llm=llm, prompt=strategy_template, verbose=True, output_key='strategies', memory=strategy_memory)

# Google SerpAPI
search = GoogleSerperAPIWrapper(api_key=os.getenv('SERPER_API_KEY'))

# Define the tools for the Langchain agent
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="Useful for when you need to ask with search."
    )
]

# Initialize the Langchain agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Define investment-related system messages
investment_system_messages = [
    SystemMessage(content="You are a friendly Stock/Financial analyst that can provide investment advice."),
    SystemMessage(content="Investing involves risks. It's important to do thorough research and consider professional advice."),
    SystemMessage(content="I can provide general information about investment strategies and concepts.")
]

# Show generated strategies if there's a prompt
if prompt:
    # Construct the conversation
    conversation = investment_system_messages + [{'content': prompt, 'role': 'user', 'content_type': 'text', 'metadata': {}}]

    # Get the answer from the LLMChain
    try:
        outputs = agent.run(conversation)
        if isinstance(outputs, dict) and strategy_chain.output_key in outputs:
            strategies = outputs[strategy_chain.output_key]
            if isinstance(strategies, list):
                for strategy in strategies:
                    st.write(strategy)
            else:
                st.write(strategies)
        else:
            st.error("No strategies generated.")
    except Exception as e:
        st.error(f"Error: {e}")

    with st.expander('Strategies History'):
        st.info(strategy_memory.buffer)
