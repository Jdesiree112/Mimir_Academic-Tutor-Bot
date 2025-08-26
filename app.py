import gradio as gr
from graph_tool import generate_plot
from metrics import EduBotMetrics
import os
import time
import logging
import json
import re
import requests
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage
from langchain.llms.base import LLM
from typing import Optional, List, Any, Type
from pydantic import BaseModel, Field
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import soundfile as sf
import atexit
import glob

# --- Environment and Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Support both token names for flexibility
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    logger.warning("Neither HF_TOKEN nor HUGGINGFACEHUB_API_TOKEN is set, the application may not work.")

metrics_tracker = EduBotMetrics(save_file="edu_metrics.json")

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
SYSTEM_PROMPT = """You are EduBot, an expert multi-concept tutor designed to facilitate genuine learning and understanding. Your primary mission is to guide students through the learning process rather than providing direct answers to academic work.

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

class Qwen25OmniLLM(LLM):
    model: Any = None
    processor: Any = None
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-Omni-7B"):
        super().__init__()
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="auto"
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Implementation for text-only responses
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)
        
        text_ids = self.model.generate(**inputs, return_audio=False)
        response = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response

    @property
    def _llm_type(self) -> str:
        return "qwen25_omni"
    
def create_langchain_agent():
    # Replace HuggingFaceHub with custom LLM
    llm = Qwen25OmniLLM()
    
    # Rest remains the same
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

def generate_voice_response(text_response: str, voice_enabled: bool = False) -> Optional[str]:
    """Generate audio response if voice is enabled."""
    if not voice_enabled:
        return None
    
    try:
        current_agent = get_agent()
        model = current_agent.llm.model
        processor = current_agent.llm.processor

        if not hasattr(model, 'generate') or not hasattr(model.generate, '__code__'):
            logger.warning("Model may not support audio generation")
            return None
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
            {"role": "user", "content": [{"type": "text", "text": "Please read this response aloud: " + text_response}]}
        ]
        
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        
        text_ids, audio = model.generate(**inputs, speaker="Ethan")
        
        # Save audio to temporary file
        audio_path = f"temp_audio_{int(time.time())}.wav"
        sf.write(audio_path, audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
        return audio_path
        
    except Exception as e:
        logger.error(f"Error generating voice response: {e}")
        return None

def cleanup_temp_audio():
    """Clean up temporary audio files on exit."""
    for file in glob.glob("temp_audio_*.wav"):
        try:
            os.remove(file)
        except:
            pass

# Register cleanup function
atexit.register(cleanup_temp_audio)

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
<title>EduBot - AI Educational Assistant</title>
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
            # LangChain will automatically handle adding HumanMessage and AIMessage to memory
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
        
        # Debug: Check message type
        logger.info(f"Message type: {type(message)}")
        logger.info(f"Message content: {message}")
        
        try:
            metrics_tracker.log_interaction(message, "user_query", "chat_start")
            logger.info("Metrics interaction logged successfully")
        except Exception as metrics_error:
            logger.error(f"Error in metrics_tracker.log_interaction: {metrics_error}")
            logger.error(f"Metrics error type: {type(metrics_error)}")
            # Continue without metrics if this fails
        
        # Generate response with LangChain
        logger.info("About to call generate_response_with_langchain")
        try:
            response = generate_response_with_langchain(message)
            logger.info(f"Response type: {type(response)}")
            logger.info(f"Response preview: {str(response)[:200]}...")
        except Exception as langchain_error:
            logger.error(f"Error in generate_response_with_langchain: {langchain_error}")
            raise langchain_error
        
        # Log metrics with timing context
        try:
            end_time = time.time()
            timing_context = f"response_time_{end_time - start_time:.2f}s"
            metrics_tracker.log_interaction(response, "bot_response", timing_context)
        except Exception as metrics_error:
            logger.error(f"Error in final metrics logging: {metrics_error}")
            # Continue without metrics if this fails
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat_response: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return f"I apologize, but I encountered an error while processing your message: {str(e)}"

def respond_and_update(message, history, voice_enabled):
    """Main function to handle user submission."""
    if not message.strip():
        return history, "", None
    
    # Add user message to history
    history.append({"role": "user", "content": message})
    yield history, "", None

    # Generate response directly (no mock streaming)
    response = chat_response(message)
    audio_path = generate_voice_response(response, voice_enabled) if voice_enabled else None
    
    history.append({"role": "assistant", "content": response})
    yield history, "", audio_path

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
        title="EduBot", 
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
            gr.HTML('<div class="title-header"><h1>🎓 EduBot</h1></div>')
            
            # Chat Section
            with gr.Row():
                chatbot = gr.Chatbot(
                    type="messages",
                    show_copy_button=True,
                    show_share_button=False,
                    avatar_images=None,
                    elem_id="main-chatbot",
                    container=False,  # Remove wrapper
                    scale=1,
                    height="70vh"  # Explicit height instead of min_height
                )
            
            # Input Section - fixed height
            with gr.Row(elem_classes=["input-controls"]):
                msg = gr.Textbox(
                    placeholder="Ask me about math, research, study strategies, or any educational topic...",
                    show_label=False,
                    lines=4,
                    max_lines=6,
                    elem_classes=["input-textbox"],
                    container=False,
                    scale=4
                )
                with gr.Column(elem_classes=["button-column"], scale=1):
                    send = gr.Button("Send", elem_classes=["send-button"], size="sm")
                    clear = gr.Button("Clear", elem_classes=["clear-button"], size="sm")
                    voice_toggle = gr.Checkbox(label="Enable Voice (Ethan)", value=False, elem_classes=["voice-toggle"])
        
            # Add audio output component
            audio_output = gr.Audio(label="Voice Response", visible=True, autoplay=True)
            
            # Event handlers - INSIDE the Blocks context
            msg.submit(respond_and_update, [msg, chatbot, voice_toggle], [chatbot, msg, audio_output])
            send.click(respond_and_update, [msg, chatbot, voice_toggle], [chatbot, msg, audio_output])
            clear.click(clear_chat, outputs=[chatbot, msg])
            
            # Apply CSS at the very end
            gr.HTML(f'<style>{custom_css}</style>')
        
        return demo

# --- Main Execution ---
if __name__ == "__main__":
    try:
        logger.info("Starting EduBot...")
        interface = create_interface()
        interface.queue()
        interface.launch()
    except Exception as e:
        logger.error(f"Failed to launch EduBot: {e}")
        raise