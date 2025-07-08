import gradio as gr
from dotenv import load_dotenv
from typing import List, Tuple, Union
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from pydantic import BaseModel
from typing import List, Tuple
from tools import search_tool, wiki_tool, save_tool, calculator_tool, content_generator_tool, unit_converter_tool, time_tool, file_reader_tool
import re

load_dotenv()

# Define the output model
class TaskResponse(BaseModel):
    query: str
    response: str
    tools_used: list[str]
    sources: list[str] = []

# Define the LLM
llm = ChatOpenAI(
    api_key="ollama",
    model="qwen3:0.6b",
    base_url="http://localhost:11434/v1",
    temperature=0.5,
    max_tokens=1000
)

# Define the parser
parser = PydanticOutputParser(pydantic_object=TaskResponse)

# Define the prompt with chat history
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a versatile AI assistant capable of handling a wide range of tasks, including answering questions, performing calculations, generating content, conducting research, unit conversions, and time zone queries.
            Use the provided tools when necessary and tailor your response to the user's query.
            For research tasks, include sources if applicable.
            Use the conversation history to provide context-aware responses.
            **Output only a valid JSON object in the format specified below. Do not include any additional text, comments, or tags like <think> outside the JSON.**
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Debug: Verify prompt type
print(f"Prompt type: {type(prompt)}")
try:
    print(f"Input variables: {prompt.input_variables}")
except AttributeError:
    print("Input variables not directly accessible; relying on automatic inference.")

# List of all available tools
all_tools = [
    search_tool,
    wiki_tool,
    save_tool,
    calculator_tool,
    content_generator_tool,
    unit_converter_tool,
    time_tool,
    file_reader_tool
]
tool_names = ["Search", "Wiki", "Save", "Calculator", "Content Generator", "Unit Converter", "Time Zone","File Reader"]

# Initialize agent
llm = llm.bind_tools(all_tools)
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=all_tools)
agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True, return_intermediate_steps=True)

# Function to format intermediate steps
def format_intermediate_steps(steps):
    if not steps:
        return "No intermediate steps taken."
    formatted = ""
    for i, step in enumerate(steps, 1):
        action = step[0]
        observation = step[1]
        formatted += f"Step {i}:\nAction: {action.tool}\nInput: {action.tool_input}\nObservation: {observation}\n\n"
    return formatted

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)  # Find the JSON object
    if match:
        return match.group(0)  # Return only the JSON part
    return text  # Fallback to original text if no JSON is found

from langchain_core.messages import HumanMessage, AIMessage

def format_chat_history(history):
    formatted = []
    for role, message in history:
        if role == "human":
            formatted.append(HumanMessage(content=message))
        elif role == "assistant":
            formatted.append(AIMessage(content=message))
    return formatted

def run_agent(prompt_input, search_cb, wiki_cb, save_cb, calculator_cb, content_generator_cb, unit_converter_cb, time_cb, file_reader_cb, history):
    selected_tools = []
    if search_cb:
        selected_tools.append("Search")
    if wiki_cb:
        selected_tools.append("Wiki")
    if save_cb:
        selected_tools.append("Save")
    if calculator_cb:
        selected_tools.append("Calculator")
    if content_generator_cb:
        selected_tools.append("Content Generator")
    if unit_converter_cb:
        selected_tools.append("Unit Converter")
    if time_cb:
        selected_tools.append("Time Zone")
    if file_reader_cb:
        selected_tools.append("File Reader")

    formatted_history = format_chat_history(history)
    try:
        response = agent_executor.invoke({"query": prompt_input, "chat_history": formatted_history})
        thoughts = format_intermediate_steps(response["intermediate_steps"])
        json_output = extract_json(response["output"])
        structured_response = parser.parse(json_output)
        summary = f"Response: {structured_response.response}\n\nTools Used: {', '.join(structured_response.tools_used) or 'None'}\n\nSources: {', '.join(structured_response.sources) or 'None'}"
        print(f"Debug - Thoughts: {thoughts[:100]}...")
        print(f"Debug - Summary: {summary[:100]}...")
        updated_history = history + [("human", prompt_input), ("assistant", structured_response.response)]
        return thoughts, summary, updated_history
    except Exception as e:
        error_msg = f"Error during agent execution: {str(e)}\nRaw output: {response['output'] if 'response' in locals() else 'No response'}"
        print(error_msg)
        return error_msg, "No summary available due to error.", history

def clear_inputs():
    return "", False, False, False, False, False, False, False, False, []

# Define custom Gradio theme
custom_theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="gray",
    neutral_hue="slate",
    font=["inter", "helvetica"],
    radius_size="md",
    spacing_size="md",
    text_size="lg"
).set(
    body_background_fill="*neutral_50",
    block_background_fill="*neutral_100",
    button_primary_background_fill="*primary_500",
    button_primary_text_color="white",
    input_background_fill="*neutral_200",
    shadow_drop="rgba(0,0,0,0.1) 0px 4px 12px"
)

# Define the Gradio interface
with gr.Blocks(theme=custom_theme, css=".gradio-container {max-width: 1200px; margin: auto;}") as demo:
    gr.Markdown(
        """
        # GPA-Qwen Enhanced
        A versatile AI assistant for research, calculations, content generation, and more. Enter your query and select tools to get started.
        """
    )
    history = gr.State(value=[])
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Enter Your Query",
                placeholder="e.g., 'Research the history of AI' or 'Convert 10 km to miles'",
                lines=5,
                show_label=True
            )
            with gr.Accordion("Select Tools", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        search_cb = gr.Checkbox(label="Search", value=False, info="Search the web for information")
                        wiki_cb = gr.Checkbox(label="Wiki", value=False, info="Query Wikipedia for details")
                        save_cb = gr.Checkbox(label="Save", value=False, info="Save output to a text file")
                        content_generator_cb = gr.Checkbox(label="Content Generator", value=False, info="Generate creative content")
                        calculator_cb = gr.Checkbox(label="Calculator", value=False, info="Perform mathematical calculations")
                        unit_converter_cb = gr.Checkbox(label="Unit Converter", value=False, info="Convert between units")
                        time_cb = gr.Checkbox(label="Time Zone", value=False, info="Get time in a specific timezone")
                        file_reader_cb = gr.Checkbox(label="File Reader", value=False, info="Attach and Read files")
            
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")
        
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Summary"):
                    summary_output = gr.Textbox(label="Summary", lines=10, placeholder="Main output goes here...",  show_label=False)
                with gr.Tab("Thoughts"):
                    thoughts_output = gr.Textbox(label="Thoughts", lines=10, placeholder="Thoughts go here...", show_label=False)
    
    gr.Markdown(
        """
        ---
        *Powered by GPA-Qwen Enhanced | Version 1.0 *
        """,
        elem_classes=["footer"]
    )

    submit_btn.click(
        run_agent,
        inputs=[prompt_input, search_cb, wiki_cb, save_cb, calculator_cb, content_generator_cb, unit_converter_cb, time_cb, file_reader_cb, history],
        outputs=[thoughts_output, summary_output, history]
    )
    clear_btn.click(
        clear_inputs,
        inputs=[],
        outputs=[prompt_input, search_cb, wiki_cb, save_cb, calculator_cb, content_generator_cb, unit_converter_cb, time_cb, file_reader_cb, history]
    )

# Launch the app
demo.launch()