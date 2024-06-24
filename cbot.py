import gradio as gr
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import urllib.parse
from functools import lru_cache

openai = OpenAI(
    api_key="HDtCy0QDu4Xj9noGAZqoNWhOl0b4vNWJ",
    base_url="https://api.deepinfra.com/v1/openai",
)

MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
BOT_AVATAR = "OpenAI_logo.png"
EXAMPLES = [
    [{"text": "What is artificial intelligence?"}],
    [{"text": "Explain quantum computing in simple terms."}],
    [{"text": "How does climate change affect biodiversity?"}],
]

@lru_cache(maxsize=128)
def extract_text_from_webpage(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.extract()
    return soup.get_text(strip=True)

def search(term, num_results=2, lang="en", timeout=5):
    escaped_term = urllib.parse.quote_plus(str(term))
    all_results = []
    max_chars_per_page = 8000
    
    with requests.Session() as session:
        resp = session.get(
            url="https://www.google.com/search",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"},
            params={"q": escaped_term, "num": num_results, "hl": lang},
            timeout=timeout,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        result_block = soup.find_all("div", attrs={"class": "g"})
        
        for result in result_block[:num_results]:
            link = result.find("a", href=True)
            if link:
                link = link["href"]
                try:
                    webpage = session.get(link, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"})
                    webpage.raise_for_status()
                    visible_text = extract_text_from_webpage(webpage.text)
                    if len(visible_text) > max_chars_per_page:
                        visible_text = visible_text[:max_chars_per_page] + "..."
                    all_results.append({"link": link, "text": visible_text})
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching or processing {link}: {e}")
                    all_results.append({"link": link, "text": None})   
    return all_results

def format_messages(user_prompt, chat_history, web_search=False):
    messages = [{"role": "system", "content": "You are OpenGPT 4o, an AI assistant. Be helpful, concise, and informative."}]
    
    for user, assistant in chat_history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    
    if web_search:
        web_results = search(user_prompt)
        web_info = ' '.join([f"Link: {res['link']}\nText: {res['text']}\n\n" for res in web_results])
        user_prompt = f"{user_prompt}\n\nRelevant web information:\n{web_info}"
    
    messages.append({"role": "user", "content": user_prompt})
    return messages

def model_inference(user_prompt, chat_history, model_selector, decoding_strategy, temperature, max_new_tokens, repetition_penalty, top_p, web_search):
    messages = format_messages(user_prompt, chat_history, web_search)
    
    chat_completion = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=float(temperature) if decoding_strategy == "Top P Sampling" else 0,
        max_tokens=int(max_new_tokens),
        top_p=float(top_p) if decoding_strategy == "Top P Sampling" else 1,
        stream=True
    )
    
    full_response = ""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            yield full_response

chatbot = gr.Chatbot(label="OpenGPT-4o Chat", avatar_images=[None, BOT_AVATAR], show_copy_button=True)
model_selector = gr.Dropdown(choices=[MODEL_NAME], value=MODEL_NAME, label="Model", visible=False)
decoding_strategy = gr.Radio(["Greedy", "Top P Sampling"], value="Top P Sampling", label="Decoding strategy")
temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature")
max_new_tokens = gr.Slider(minimum=1, maximum=1000, value=256, step=1, label="Max new tokens")
repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.1, label="Repetition penalty")
top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="Top P")

