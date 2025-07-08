from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import math

def save_to_txt(data: str, filename: str = "output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Task Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

def calculate(expression: str) -> str:
    try:
        # Simple evaluation for basic math expressions
        result = eval(expression, {"__builtins__": {}, "math": math})
        return f"Result: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

def generate_content(prompt: str) -> str:
    # Placeholder for content generation; in practice, this could use the LLM directly
    return f"Generated content based on prompt: {prompt}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves task output to a text file.",
)

search_tool = Tool(
    name="search",
    func=DuckDuckGoSearchRun().run,
    description="Search the web for information",
)

wiki_tool = Tool(
    name="wiki",
    func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100000)).run,
    description="Query Wikipedia for information",
)

calculator_tool = Tool(
    name="calculator",
    func=calculate,
    description="Perform mathematical calculations using Python's eval with math module",
)

content_generator_tool = Tool(
    name="content_generator",
    func=generate_content,
    description="Generate creative content based on a given prompt",
)

import asyncio
from functools import wraps

async def async_search(query):
    return await asyncio.to_thread(search_tool.func, query)

async def async_wiki(query):
    return await asyncio.to_thread(wiki_tool.func, query)