from django.http import HttpResponse
from django.conf import settings
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import gradio

# Load pre-trained model and tokenizer
model_name = "gpt2"  # Use "gpt2" for GPT-2 model or specify any other model from the Hugging Face model hub
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

messages = [{"role": "system", "content": "You are a financial expert that specializes in real estate investment and negotiation"}]

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)
    ChatGPT_reply = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

demo = gradio.Interface(fn=CustomChatGPT, inputs="text", outputs="text", title="Real Estate Pro")

demo.launch(share=True)
