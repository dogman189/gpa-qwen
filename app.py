import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool, calculator_tool, content_generator_tool

load_dotenv()

# Define the LLM
llm = ChatOpenAI(
    api_key="ollama",
    model="qwen3:0.6b",
    base_url="http://localhost:11434/v1",
    temperature=0.5,
    max_tokens=1000
)

# Define the prompt without input_variables
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a versatile AI assistant capable of handling a wide range of tasks, including answering questions, performing calculations, generating content, and conducting research.
            Use the provided tools when necessary and tailor your response to the user's query.
            For research tasks, include sources if applicable.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Debug: Verify prompt type and input variables
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
    content_generator_tool
]
tool_names = ["Search", "Wiki", "Save", "Calculator", "Content Generator"]

# Function to format intermediate steps
def format_intermediate_steps(steps):
    formatted = ""
    for i, step in enumerate(steps, 1):
        action = step[0]
        observation = step[1]
        formatted += f"Step {i}:\nAction: {action.tool}\nInput: {action.tool_input}\nObservation: {observation}\n\n"
    return formatted

# Function to run the agent
def run_agent(prompt_input, search_enabled, wiki_enabled, save_enabled, calculator_enabled, content_generator_enabled):
    selected_tools = []
    if search_enabled:
        selected_tools.append(search_tool)
    if wiki_enabled:
        selected_tools.append(wiki_tool)
    if save_enabled:
        selected_tools.append(save_tool)
    if calculator_enabled:
        selected_tools.append(calculator_tool)
    if content_generator_enabled:
        selected_tools.append(content_generator_tool)
    
    llm_bound = llm.bind_tools(selected_tools)
    agent = create_tool_calling_agent(llm=llm_bound, prompt=prompt, tools=selected_tools)
    agent_executor = AgentExecutor(agent=agent, tools=selected_tools, verbose=True, return_intermediate_steps=True)
    
    try:
        response = agent_executor.invoke({"query": prompt_input})
        thoughts = format_intermediate_steps(response["intermediate_steps"])
        summary = response["output"]
        return thoughts, summary
    except Exception as e:
        return f"Error during agent execution: {str(e)}", "No summary available due to error."

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Assistant")
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Enter your prompt", placeholder="Type your query here...")
            with gr.Row():
                search_cb = gr.Checkbox(label="Search", value=False)
                wiki_cb = gr.Checkbox(label="Wiki", value=False)
                save_cb = gr.Checkbox(label="Save", value=False)
                calculator_cb = gr.Checkbox(label="Calculator", value=False)
                content_generator_cb = gr.Checkbox(label="Content Generator", value=False)
            submit_btn = gr.Button("Submit")
        with gr.Column():
            thoughts_output = gr.Textbox(label="Thoughts", lines=10)
            summary_output = gr.Textbox(label="Summary", lines=10)
    
    submit_btn.click(
        run_agent,
        inputs=[prompt_input, search_cb, wiki_cb, save_cb, calculator_cb, content_generator_cb],
        outputs=[thoughts_output, summary_output]
    )

# Launch the app
demo.launch()