from django.http import HttpResponse  # Import the HttpResponse class from the django.http module

#from django.conf import settings  # Import the settings module from Django configuration
#from decouple import config  # Import the config function from the decouple module

from transformers import GPT2LMHeadModel, GPT2Tokenizer  # Import GPT2LMHeadModel and GPT2Tokenizer classes from the transformers module
import openai  # Import the openai module

#import torch  # Import the torch module
import gradio  # Import the gradio module

# Load pre-trained model and tokenizer
openai.api_key = '####'  # Set the OpenAI API key for authentication

messages = [{"role": "system", "content": "You are a financial experts that specializes in real estate investment and negotiation"}]  # Initialize a list of messages with a system message

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})  # Append the user input to the messages list
    response = openai.ChatCompletion.create(  # Call the ChatCompletion.create method from the openai module
        model="gpt-3.5-turbo",  # Specify the model name as "gpt-3.5-turbo"
        messages=messages  # Pass the messages list as input to the model
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]  # Extract the assistant's reply from the response
    messages.append({"role": "assistant", "content": ChatGPT_reply})  # Append the assistant's reply to the messages list
    return ChatGPT_reply  # Return the assistant's reply

demo = gradio.Interface(fn=CustomChatGPT, inputs="text", outputs="text", title="Real Estate Pro")  # Create a gradio Interface object with CustomChatGPT function

demo.launch(share=True)  # Launch the gradio interface and enable sharing
