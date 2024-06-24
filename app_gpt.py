import gradio as gr
from cbot import chatbot, model_inference, BOT_AVATAR, EXAMPLES, model_selector, decoding_strategy, temperature, max_new_tokens, repetition_penalty, top_p

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

def create_gradio_blocks():
    with gr.Blocks(
            fill_height=True,
            css=""".gradio-container .avatar-container {height: 40px width: 40px !important;} #duplicate-button {margin: auto; color: white; background: #f1a139; border-radius: 100vh; margin-top: 2px; margin-bottom: 2px;}""",
        theme=theme,
        title="OpenGPT 4o DEMO"
    ) as demo:
        gr.Markdown("# OpenGPT 4o")
        gr.Markdown("### Chat with OpenGPT 4o")
        with gr.Row(elem_id="model_selector_row"):
            pass  
    
        decoding_strategy.change(
            fn=lambda selection: gr.Slider(
                visible=(
                    selection
                    in [
                        "contrastive_sampling",
                        "beam_sampling",
                        "Top P Sampling",
                        "sampling_top_k",
                    ]
                )
            ),
            inputs=decoding_strategy,
            outputs=temperature,
        )

        decoding_strategy.change(
            fn=lambda selection: gr.Slider(visible=(selection in ["Top P Sampling"])),
            inputs=decoding_strategy,
            outputs=top_p,
        )
        
        gr.ChatInterface(
            fn=model_inference,
            chatbot=chatbot,
            examples=EXAMPLES,
            multimodal=True,
            cache_examples=False,
            additional_inputs=[
            model_selector,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
            # gr.Checkbox(label="Web Search", value=True),
            ],
        )    
    return demo

gradio_app = create_gradio_blocks()