import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

messages = [{"role": "system", "content": "I am your health specialist. Ask me anything you wish"}]

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)  # Set attention mask to 1 for all input tokens
    pad_token_id = tokenizer.eos_token_id  # Set pad token ID to end-of-sequence token ID
    output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=pad_token_id, attention_mask=attention_mask)
    ChatGPT_reply = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

def main():
    st.title("Health Speacialist")
    user_input = st.text_input("Enter your message:")
    if st.button("Send"):
        if user_input:
            response = CustomChatGPT(user_input)
            st.write("Assistant's Reply:")
            st.write(response)
            # Optionally, you can display the conversation history
            st.write("Conversation History:")
            for message in messages:
                st.write(f"{message['role']}: {message['content']}")
        else:
            st.warning("Please enter a message.")

if __name__ == "__main__":
    main()
