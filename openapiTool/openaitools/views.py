import openai  # Import the openai module. Openai is a module that allows us to efficiently communicate with the openapikey

import gradio  # Import the gradio module. Gradio is used as a template for deploying our model after it is generated

# Load pre-trained model and tokenizer
openai.api_key = '####'  # Set the OpenAI API key for authentication.

messages = [{"role": "system", "content": "You are a financial experts that specializes in real estate investment and negotiation"}]  # Initialize a list of messages with a system message

def CustomChatGPT(user_input): #Define a function that allows users to efficiently input 
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
