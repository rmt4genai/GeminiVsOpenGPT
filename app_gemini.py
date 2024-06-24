import os
import time
from typing import List, Tuple, Optional, Dict, Union
from dotenv import load_dotenv
import google.generativeai as genai
import gradio as gr
import asyncio
print("google-generativeai:", genai.__version__)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

BOT_AVATAR = "https://media.roboflow.com/spaces/gemini-icon.png"
EXAMPLES = [
    [{"text": "What is artificial intelligence?"}],
    [{"text": "Explain quantum computing in simple terms."}],
    [{"text": "How does climate change affect biodiversity?"}],
]

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="orange",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont('Libre Franklin'), gr.themes.GoogleFont('Public Sans'), 'system-ui', 'sans-serif']
).set(
    body_background_fill_dark="#111111",
    block_background_fill_dark="#111111",
    block_border_width="1px",
    block_title_background_fill_dark="#1e1c26",
    input_background_fill_dark="#292733",
    button_secondary_background_fill_dark="#24212b",
    border_color_primary_dark="#343140",
    background_fill_secondary_dark="#111111",
    color_accent_soft_dark="transparent"
)

def preprocess_stop_sequences(stop_sequences: str) -> Optional[List[str]]:
    if not stop_sequences:
        return None
    return [sequence.strip() for sequence in stop_sequences.split(",")]

def preprocess_chat_history(history: List[List[Optional[str]]]) -> List[dict]:
    messages = []
    for user_message, model_message in history:
        if user_message:
            messages.append({"role": "user", "parts": [user_message]})
        if model_message:
            messages.append({"role": "model", "parts": [model_message]})
    return messages
# def preprocess_chat_history(
#     history: List[Tuple[Optional[Union[Tuple[str], str]], Optional[str]]]
# ) -> List[Dict[str, Union[str, List[str]]]]:
#     messages = []
#     for user_message, model_message in history:
#         if isinstance(user_message, tuple):
#             pass
#         elif user_message is not None:
#             messages.append({'role': 'user', 'parts': [user_message]})
#         if model_message is not None:
#             messages.append({'role': 'model', 'parts': [model_message]})
#     return messages

def user(text_prompt: str, chatbot: List[Tuple[Optional[Union[Tuple[str], str]], Optional[str]]]):
    if text_prompt:
        chatbot.append((text_prompt, None))
    return "", chatbot

async def bot(chatbot: List[Tuple[Optional[Union[Tuple[str], str]], Optional[str]]],temperature: float, max_output_tokens: int, top_p: float):
    if len(chatbot) == 0:
        yield chatbot

    messages = preprocess_chat_history(chatbot)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content(
        messages,
        stream=True,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p
        )
    )
        
    chatbot[-1][1] = ""
    for chunk in response:
        for i in range(0, len(chunk.text), 10):
            section = chunk.text[i:i + 10]
            chatbot[-1][1] += section
            await asyncio.sleep(0.01)
            yield chatbot
            

def create_gradio_blocks():
    with gr.Blocks(
        fill_height=True,
        css=""".gradio-container .avatar-container {height: 40px width: 40px !important;} #duplicate-button {margin: auto; color: white; background: #f1a139; border-radius: 100vh; margin-top: 2px; margin-bottom: 2px;}""",
        theme=theme,
        title="Gemini 1.5 Pro Playground"
    ) as demo:
        gr.Markdown("# Gemini 1.5 Pro Playground 💬")
        gr.Markdown("### Play with Gemini 1.5 Pro")
        
        chatbot_component = gr.Chatbot(label='Gemini', avatar_images=[None, BOT_AVATAR], show_copy_button=True)
        text_prompt_component = gr.Textbox(placeholder="Hi there! [press Enter]", show_label=False, autofocus=True)
        run_button_component = gr.Button(value="Run", variant="primary")
        
        with gr.Column():
            chatbot_component
            with gr.Row():
                text_prompt_component
                run_button_component
                
        user_inputs = [
            text_prompt_component,
            chatbot_component
        ]

        bot_inputs = [
            chatbot_component
        ]

        run_button_component.click(
            fn=user,
            inputs=user_inputs,
            outputs=[text_prompt_component, chatbot_component],
            queue=False
        ).then(
            fn=bot, inputs=bot_inputs, outputs=[chatbot_component],
        )

        text_prompt_component.submit(
            fn=user,
            inputs=user_inputs,
            outputs=[text_prompt_component, chatbot_component],
            queue=False
        ).then(
            fn=bot, inputs=bot_inputs, outputs=[chatbot_component],
        )

    return demo

gradio_app_demo = create_gradio_blocks()


# import os
# import time
# import uuid
# from typing import List, Tuple, Optional, Dict, Union
# from dotenv import load_dotenv
# import google.generativeai as genai
# import gradio as gr

# print("google-generativeai:", genai.__version__)

# # Load environment variables from .env file
# load_dotenv()

# # Get the API key from the environment variable
# GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# if not GOOGLE_API_KEY:
#     raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

# # Configure the genai library with the API key
# genai.configure(api_key=GOOGLE_API_KEY)

# TITLE = """<h1 align="center">Gemini 1.5 Pro Playground 💬</h1>"""
# SUBTITLE = """<h2 align="center">Play with Gemini 1.5 Pro</h2>"""

# AVATAR_IMAGES = (
#     None,
#     "https://media.roboflow.com/spaces/gemini-icon.png"
# )

# CHAT_HISTORY = List[Tuple[Optional[Union[Tuple[str], str]], Optional[str]]]

# def preprocess_stop_sequences(stop_sequences: str) -> Optional[List[str]]:
#     if not stop_sequences:
#         return None
#     return [sequence.strip() for sequence in stop_sequences.split(",")]

# def preprocess_chat_history(
#     history: CHAT_HISTORY
# ) -> List[Dict[str, Union[str, List[str]]]]:
#     messages = []
#     for user_message, model_message in history:
#         if isinstance(user_message, tuple):
#             pass
#         elif user_message is not None:
#             messages.append({'role': 'user', 'parts': [user_message]})
#         if model_message is not None:
#             messages.append({'role': 'model', 'parts': [model_message]})
#     return messages

# def user(text_prompt: str, chatbot: CHAT_HISTORY):
#     if text_prompt:
#         chatbot.append((text_prompt, None))
#     return "", chatbot

# def bot(
#     chatbot: CHAT_HISTORY
# ):
#     if len(chatbot) == 0:
#         return chatbot

#     messages = preprocess_chat_history(chatbot)
#     model = genai.GenerativeModel('gemini-1.5-pro-latest')
#     response = model.generate_content(
#         messages,
#         stream=True,
#         )
        
#     chatbot[-1][1] = ""
#     for chunk in response:
#         for i in range(0, len(chunk.text), 10):
#             section = chunk.text[i:i + 10]
#             chatbot[-1][1] += section
#             time.sleep(0.01)
#             yield chatbot

# chatbot_component = gr.Chatbot(
#     label='Gemini',
#     bubble_full_width=False,
#     avatar_images=AVATAR_IMAGES,
#     scale=2,
#     height=400
# )
# text_prompt_component = gr.Textbox(
#     placeholder="Hi there! [press Enter]", show_label=False, autofocus=True, scale=8
# )
# run_button_component = gr.Button(value="Run", variant="primary", scale=1)

# user_inputs = [
#     text_prompt_component,
#     chatbot_component
# ]

# bot_inputs = [
#     # run_button_component,
#     chatbot_component
# ]

# with gr.Blocks() as demo:
#     gr.HTML(TITLE)
#     gr.HTML(SUBTITLE)
    
#     with gr.Column():
#         chatbot_component.render()
#         with gr.Row():
#             text_prompt_component.render()
#             run_button_component.render()

#     run_button_component.click(
#         fn=user,
#         inputs=user_inputs,
#         outputs=[text_prompt_component, chatbot_component],
#         queue=False
#     ).then(
#         fn=bot, inputs=bot_inputs, outputs=[chatbot_component],
#     )

#     text_prompt_component.submit(
#         fn=user,
#         inputs=user_inputs,
#         outputs=[text_prompt_component, chatbot_component],
#         queue=False
#     ).then(
#         fn=bot, inputs=bot_inputs, outputs=[chatbot_component],
#     )

# # Expose the Gradio interface
# gradio_app_demo = demo
