# Imports
import os 

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# setup LLM and Tools
from langchain_groq import ChatGroq
# from langchain_openai  import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# openai_llm=ChatOpenAI(model="gpt-4o-mini")
groq_llm=ChatGroq(model="llama-3.3-70b-versatile")

search_tool=TavilySearchResults(max_results=2)

# setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt = "Act as an A chatbot who is smart and friendly."

agent=create_react_agent(
    model=groq_llm,
    tools=[search_tool],
    state_modifier=system_prompt
)

query = "Tell me about the trends in crypto markets"
state = {"message": query}
response = agent.invoke(state)
messages = response.get("messages")
ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
print(ai_messages[-1])
