import gradio as gr
from graph_tool import generate_plot
from metrics import MimirMetrics
import os
import time
from dotenv import load_dotenv
import logging
import json
import re
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage
from langchain.llms.base import LLM
from typing import Optional, List, Any, Type
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# Load environment variables from .EVN fil (case-sensitive)
load_dotenv(".evn")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
print("Environment variables loaded.")

# --- Environment and Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Support both token names for flexibility
hf_token = HF_TOKEN
if not hf_token:
    logger.warning("Neither HF_TOKEN nor HUGGINGFACEHUB_API_TOKEN is set, the application may not work.")

metrics_tracker = MimirMetrics(save_file="Mimir_metrics.json")

# --- LangChain Tool Definition ---
class GraphInput(BaseModel):
    data_json: str = Field(description="JSON string of data for the graph")
    labels_json: str = Field(description="JSON string of labels for the graph", default="[]")
    plot_type: str = Field(description="Type of plot: bar, line, or pie")
    title: str = Field(description="Title for the graph")
    x_label: str = Field(description="X-axis label", default="")
    y_label: str = Field(description="Y-axis label", default="")

class CreateGraphTool(BaseTool):
    name: str = "create_graph"
    description: str = """Generates a plot (bar, line, or pie) and returns it as an HTML-formatted Base64-encoded image string. Use this tool when teaching concepts that benefit from visual representation, such as: statistical distributions, mathematical functions, data comparisons, survey results, grade analyses, scientific relationships, economic models, or any quantitative information that would be clearer with a graph. 

REQUIRED FORMAT:
- data_json: A JSON dictionary where keys are category names and values are numbers
  Example: '{"Math": 85, "Science": 92, "English": 78}'
- labels_json: A JSON list, only needed for pie charts if you want custom labels different from the data keys. For bar/line charts, use empty list: '[]'
  Example for pie: '["Mathematics", "Science", "English Literature"]'
  Example for bar/line: '[]'

EXAMPLES:
Bar chart: data_json='{"Q1": 1000, "Q2": 1200, "Q3": 950}', labels_json='[]'
Line chart: data_json='{"Jan": 100, "Feb": 120, "Mar": 110}', labels_json='[]' 
Pie chart: data_json='{"A": 30, "B": 45, "C": 25}', labels_json='["Category A", "Category B", "Category C"]'

Always use proper JSON formatting with quotes around keys and string values."""
    args_schema: Type[BaseModel] = GraphInput
    
    def _run(self, data_json: str, labels_json: str = "[]", plot_type: str = "bar", 
             title: str = "Chart", x_label: str = "", y_label: str = "") -> str:
        try:
            # Validate JSON format before passing to generate_plot
            import json
            try:
                data_parsed = json.loads(data_json)
                labels_parsed = json.loads(labels_json)
                
                # Validate data structure
                if not isinstance(data_parsed, dict):
                    return "<p style='color:red;'>data_json must be a JSON dictionary with string keys and numeric values.</p>"
                    
                if not isinstance(labels_parsed, list):
                    return "<p style='color:red;'>labels_json must be a JSON list (use [] if no custom labels needed).</p>"
                    
            except json.JSONDecodeError as json_error:
                return f"<p style='color:red;'>Invalid JSON format: {str(json_error)}. Ensure proper JSON formatting with quotes.</p>"
            
            return generate_plot(
                data_json=data_json,
                labels_json=labels_json,
                plot_type=plot_type,
                title=title,
                x_label=x_label,
                y_label=y_label
            )
        except Exception as e:
            return f"<p style='color:red;'>Error creating graph: {str(e)}</p>"


# --- System Prompt ---
SYSTEM_PROMPT = """You are Mimir, an expert multi-concept tutor designed to facilitate genuine learning and understanding. Your primary mission is to guide students through the learning process rather than providing direct answers to academic work.

## Core Educational Principles
- Provide comprehensive, educational responses that help students truly understand concepts
- Use minimal formatting, with markdown bolding reserved for **key terms** only
- Prioritize teaching methodology over answer delivery
- Foster critical thinking and independent problem-solving skills

## Tone and Communication Style
- Maintain an engaging, friendly tone appropriate for high school students
- Write at a reading level that is accessible yet intellectually stimulating
- Be supportive and encouraging without being condescending
- Never use crude language or content inappropriate for an educational setting
- Avoid preachy, judgmental, or accusatory language
- Skip flattery and respond directly to questions
- Do not use emojis or actions in asterisks unless specifically requested
- Present critiques and corrections kindly as educational opportunities

## Academic Integrity Approach
You recognize that students may seek direct answers to homework, assignments, or test questions. Rather than providing complete solutions or making accusations about intent, you should:

- **Guide through processes**: Break down problems into conceptual components and teach underlying principles
- **Ask clarifying questions**: Understand what the student already knows and where their confusion lies
- **Provide similar examples**: Work through analogous problems that demonstrate the same concepts without directly solving their specific assignment
- **Encourage original thinking**: Help students develop their own reasoning and analytical skills
- **Suggest study strategies**: Recommend effective learning approaches for the subject matter

## Tool Usage
You have access to a create_graph tool. Use this tool naturally when a visual representation would enhance understanding or when discussing concepts that involve data, relationships, patterns, or quantitative information. Consider creating graphs for:
- Mathematical concepts (functions, distributions, relationships)
- Statistical examples and explanations
- Scientific data and relationships
- Practice problems involving graph interpretation
- Comparative analyses
- Economic models or business concepts
- Any situation where visualization aids comprehension

When using the create_graph tool, format data as JSON strings:
- data_json: '{"Category1": 25, "Category2": 40, "Category3": 35}'
- labels_json: '["Category1", "Category2", "Category3"]'

## Response Guidelines
- **For math problems**: Explain concepts, provide formula derivations, and guide through problem-solving steps without computing final numerical answers
- **For multiple-choice questions**: Discuss the concepts being tested and help students understand how to analyze options rather than identifying the correct choice
- **For essays or written work**: Discuss research strategies, organizational techniques, and critical thinking approaches rather than providing content or thesis statements
- **For factual questions**: Provide educational context and encourage students to synthesize information rather than stating direct answers
- Use graphs naturally when they would clarify or enhance your explanations

## Communication Guidelines
- Maintain a supportive, non-judgmental tone in all interactions
- Assume positive intent while redirecting toward genuine learning
- Use Socratic questioning to promote discovery and critical thinking
- Celebrate understanding and progress in the learning process
- Encourage students to explain their thinking and reasoning
- Provide honest, accurate feedback even when it may not be what the student wants to hear

Your goal is to be an educational partner who empowers students to succeed through understanding, not a service that completes their work for them."""

# --- Improved LangChain Setup ---
# Global flag to track system prompt initialization
system_prompt_initialized = False

def initialize_system_prompt(agent):
    """Initialize the system prompt as a SystemMessage in memory."""
    global system_prompt_initialized
    if not system_prompt_initialized:
        system_message = SystemMessage(content=SYSTEM_PROMPT)
        agent.memory.chat_memory.add_message(system_message)
        system_prompt_initialized = True

class Qwen25SmallLLM(LLM):
    model: Any = None
    tokenizer: Any = None
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-3B-Instruct"):
        super().__init__()
        logger.info(f"Loading model: {model_path}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        logger.info("Model loaded successfully")
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate text response using the local model."""
        try:
            # Format the conversation
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            model_inputs = self.tokenizer([text], return_tensors="pt")
            if torch.cuda.is_available():
                model_inputs = model_inputs.to(self.model.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=1000,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in model generation: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "qwen25_small"
    
def create_langchain_agent():
    # Use the smaller local model
    llm = Qwen25SmallLLM()
    
    tools = [CreateGraphTool()]
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=10,
        return_messages=True
    )
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        max_iterations=3,
        early_stopping_method="generate"
    )
    
    return agent

# --- Global Agent Instance ---
agent = None

def get_agent():
    """Get or create the LangChain agent."""
    global agent
    if agent is None:
        agent = create_langchain_agent()
    return agent

# --- UI: MathJax Configuration ---
mathjax_config = '''
<script>
window.MathJax = {
  tex: {
    inlineMath: [['\\\\(', '\\\\)']],
    displayMath: [['$', '$'], ['\\\\[', '\\\\]']],
    packages: {'[+]': ['ams']}
  },
  svg: {fontCache: 'global'},
  startup: {
    ready: () => {
      MathJax.startup.defaultReady();
      // Re-render math when new content is added
      const observer = new MutationObserver(function(mutations) {
        MathJax.typesetPromise();
      });
      observer.observe(document.body, {childList: true, subtree: true});
    }
  }
};
</script>
'''

# --- HTML Head Content ---
html_head_content = '''
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mimir - AI Educational Assistant</title>
'''

# --- Force Light Mode Script ---
force_light_mode = '''
<script>
// Force light theme in Gradio
window.addEventListener('DOMContentLoaded', function () {
    const gradioURL = window.location.href;
    const url = new URL(gradioURL);
    const currentTheme = url.searchParams.get('__theme');
    
    if (currentTheme !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.replace(url.toString());
    }
});
</script>
'''

# --- Core Logic Functions ---
def smart_truncate(text, max_length=3000):
    """Truncates text intelligently to the last full sentence or word."""
    if len(text) <= max_length:
        return text
    
    # Try to split by sentence
    sentences = re.split(r'(?<=[.!?])\s+', text[:max_length])
    if len(sentences) > 1:
        return ' '.join(sentences[:-1]) + "... [Response truncated - ask for continuation]"
    # Otherwise, split by word
    words = text[:max_length].split()
    return ' '.join(words[:-1]) + "... [Response truncated]"

def generate_response_with_langchain(message, max_retries=3):
    """Generate response using LangChain agent with proper message handling."""
    
    for attempt in range(max_retries):
        try:
            # Get the agent
            current_agent = get_agent()
            
            # Initialize system prompt if not already done
            initialize_system_prompt(current_agent)
            
            # Use the agent directly with the message
            response = current_agent.run(input=message)
            
            return smart_truncate(response)
            
        except Exception as e:
            logger.error(f"LangChain error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                return f"I apologize, but I encountered an error while processing your message: {str(e)}"

def chat_response(message, history=None):
    """Process chat message and return response."""
    try:
        # Track metrics with timing context
        start_time = time.time()
        timing_context = {
            'start_time': start_time,
            'chunk_count': 0,
            'provider_latency': 0.0
        }
        
        try:
            # Log start of interaction
            metrics_tracker.log_interaction(
                query=message,
                response="", 
                timing_context=timing_context,
                error_occurred=False
            )
            logger.info("Metrics interaction logged successfully")
        except Exception as metrics_error:
            logger.error(f"Error in metrics_tracker.log_interaction: {metrics_error}")
        
        # Generate response with LangChain
        response = generate_response_with_langchain(message)
        
        # Log final metrics
        try:
            metrics_tracker.log_interaction(
                query=message,
                response=response,
                timing_context=timing_context,
                error_occurred=False
            )
        except Exception as metrics_error:
            logger.error(f"Error in final metrics logging: {metrics_error}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat_response: {e}")
        return f"I apologize, but I encountered an error while processing your message: {str(e)}"

def respond_and_update(message, history):
    """Main function to handle user submission - no voice parameter."""
    if not message.strip():
        return history, ""
    
    # Add user message to history
    history.append({"role": "user", "content": message})
    yield history, ""

    # Generate response
    response = chat_response(message)
    
    history.append({"role": "assistant", "content": response})
    yield history, ""

def clear_chat():
    """Clear the chat history and reset system prompt flag."""
    global agent, system_prompt_initialized
    if agent is not None:
        agent.memory.clear()
    system_prompt_initialized = False
    return [], ""

# --- UI: Interface Creation ---
def create_interface():
    """Creates and configures the complete Gradio interface."""
    
    # Read CSS file
    custom_css = ""
    try:
        with open("styles.css", "r", encoding="utf-8") as css_file:
            custom_css = css_file.read()
    except FileNotFoundError:
        logger.warning("styles.css file not found, using default styling")
    except Exception as e:
        logger.warning(f"Error reading styles.css: {e}")
    
    with gr.Blocks(
        title="Mimir", 
        fill_width=True, 
        fill_height=True,
        theme=gr.themes.Origin()
    ) as demo:
        # Add head content and MathJax
        gr.HTML(html_head_content)
        gr.HTML(force_light_mode)
        gr.HTML('<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>')
        gr.HTML(mathjax_config)
        
        with gr.Column(elem_classes=["main-container"]):
            # Title Section
            gr.HTML('<div class="title-header"><h1> Mimir 🎓</h1></div>')
            
            # Chat Section
            with gr.Row():
                chatbot = gr.Chatbot(
                    type="messages",
                    show_copy_button=True,
                    show_share_button=False,
                    avatar_images=None,
                    elem_id="main-chatbot",
                    container=False,
                    scale=1,
                    height="70vh"
                )
            
            # Input Section
            with gr.Row(elem_classes=["input-controls"]):
                msg = gr.Textbox(
                    placeholder="Ask me about math, research, study strategies, or any educational topic...",
                    show_label=False,
                    lines=6,
                    max_lines=8,
                    elem_classes=["input-textbox"],
                    container=False,
                    scale=4
                )
                with gr.Column(elem_classes=["button-column"], scale=1):
                    send = gr.Button("Send", elem_classes=["send-button"], size="sm")
                    clear = gr.Button("Clear", elem_classes=["clear-button"], size="sm")
            
            # Event handlers - no voice parameter
            msg.submit(respond_and_update, [msg, chatbot], [chatbot, msg])
            send.click(respond_and_update, [msg, chatbot], [chatbot, msg])
            clear.click(clear_chat, outputs=[chatbot, msg])
            
            # Apply CSS at the very end
            gr.HTML(f'<style>{custom_css}</style>')
        
        return demo

# --- Main Execution ---
if __name__ == "__main__":
    try:
        logger.info("Starting Mimir...")
        interface = create_interface()
        interface.queue()
        interface.launch(
            server_name="0.0.0.0",
            share=True,
            debug=True,
            favicon_path="assets/favicon.ico"
        )
    except Exception as e:
        logger.error(f"Failed to launch Mimir: {e}")
        raise