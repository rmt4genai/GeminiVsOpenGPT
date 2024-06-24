import gradio as gr
from app_gemini import bot as gemini_bot, preprocess_chat_history as gemini_preprocess
from app_gpt import model_inference as gpt_inference
from cbot import MODEL_NAME, BOT_AVATAR as GPT_AVATAR
from typing import List, Optional, Union
import asyncio
from app_gemini import theme
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_combined_interface():
    with gr.Blocks(theme=theme, title="Play with Gemini and GPT") as demo:
        gr.Markdown("# Play with Gemini and GPT")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Gemini 1.5 Pro")
                gemini_chatbot = gr.Chatbot(label='Gemini', avatar_images=[None, "https://media.roboflow.com/spaces/gemini-icon.png"], show_copy_button=True)
            
            with gr.Column():
                gr.Markdown("## OpenGPT 4o")
                gpt_chatbot = gr.Chatbot(label="OpenGPT-4o Chat", avatar_images=[None, GPT_AVATAR], show_copy_button=True)

        text_input = gr.Textbox(placeholder="Enter your prompt here...", label="Input")
        send_button = gr.Button("RUN")

        # GPT model parameters
        with gr.Accordion("LLM Model Parameters", open=False):
            gpt_model_selector = gr.Dropdown(choices=[MODEL_NAME], value=MODEL_NAME, label="Model")
            decoding_strategy = gr.Radio(["Greedy", "Top P Sampling"], value="Top P Sampling", label="Decoding strategy")
            temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature")
            max_new_tokens = gr.Slider(minimum=1, maximum=1000, value=512, step=1, label="Max new tokens")
            repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.1, label="Repetition penalty")
            top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="Top P")
            web_search = gr.Checkbox(label="Web Search", value=False)

        def process_input(message: str, gemini_history: List[List[Optional[str]]], gpt_history: List[List[Optional[str]]]):
            gemini_history.append([message, None])
            gemini_messages = gemini_preprocess(gemini_history)
            gpt_history.append([message, None])
            return gemini_messages, gemini_history, gpt_history

        async def update_both_chatbots(message, gemini_history, gpt_history, model, decoding_strategy, temperature, max_new_tokens, repetition_penalty, top_p, web_search):
            gemini_messages, gemini_history, gpt_history = process_input(message, gemini_history, gpt_history)
            
            # Generate Gemini response
            try:
                async for chunk in gemini_bot(gemini_history,temperature=temperature, max_output_tokens=max_new_tokens, top_p=top_p):
                    yield chunk, gpt_history, ""
            except Exception as e:
                gemini_history[-1][1] = f"An error occurred with Gemini: {str(e)}"
                yield gemini_history, gpt_history, ""

            # Generate GPT response
            try:
                for chunk in gpt_inference(message, gpt_history[:-1], model, decoding_strategy, temperature, max_new_tokens, repetition_penalty, top_p, web_search):
                    gpt_history[-1][1] = chunk
                    yield gemini_history, gpt_history, ""
            except Exception as e:
                gpt_history[-1][1] = f"An error occurred with GPT: {str(e)}"
                yield gemini_history, gpt_history, ""
        
        send_button.click(
            fn=update_both_chatbots,
            inputs=[text_input, gemini_chatbot, gpt_chatbot, gpt_model_selector, decoding_strategy, temperature, max_new_tokens, repetition_penalty, top_p, web_search],
            outputs=[gemini_chatbot, gpt_chatbot,text_input]
        ).then(lambda: None)  # Add this to handle the async function

        text_input.submit(
            fn=update_both_chatbots,
            inputs=[text_input, gemini_chatbot, gpt_chatbot, gpt_model_selector, decoding_strategy, temperature, max_new_tokens, repetition_penalty, top_p, web_search],
            outputs=[gemini_chatbot, gpt_chatbot,text_input]
        ).then(lambda: None)

    return demo

gradio_app = create_combined_interface()

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, gradio_app, path="/")

# Add a simple FastAPI endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)