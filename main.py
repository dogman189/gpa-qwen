from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool, calculator_tool, content_generator_tool

load_dotenv()

class TaskResponse(BaseModel):
    query: str
    response: str
    tools_used: list[str]
    sources: list[str] = []  # Optional, only for research-related tasks

llm = ChatOpenAI(
    api_key="ollama",
    model="qwen3:0.6b",
    base_url="http://localhost:11434/v1",
    temperature=0.5,  # Lower for precise tasks, adjustable for creativity
    max_tokens=1000   # Increased for more complex responses
)

parser = PydanticOutputParser(pydantic_object=TaskResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a versatile AI assistant capable of handling a wide range of tasks, including answering questions, performing calculations, generating content, and conducting research.
            Use the provided tools when necessary and tailor your response to the user's query.
            For research tasks, include sources if applicable.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [
    search_tool,
    wiki_tool,
    save_tool,
    calculator_tool,
    content_generator_tool
]

llm = llm.bind_tools(tools)
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
query = input("How can I assist you today? ")
raw_response = agent_executor.invoke({"query": query})

try:
    text = raw_response["output"]
    structured_response = parser.parse(text)
    print(structured_response)
except Exception as e:
    print("End of response")