import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Sidebar for model size
st.sidebar.title("⚙️ Settings")
model_size = st.sidebar.selectbox("Select Model Size", ["gpt2", "gpt2-medium", "gpt2-large"])

# Load model and tokenizer
@st.cache_resource
def load_model(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model(model_size)

# Task-based prompt builder
def build_prompt(task, user_input):
    if task == "Email":
        return (
            "Write a professional email with the following details:\n"
            f"Subject: {user_input}\n"
            "Include a proper greeting, body, and closing.\n\nEmail:\n"
        )
    elif task == "Essay":
        return (
            f"Write a detailed essay on the topic: '{user_input}'.\n"
            "Include an introduction, body, and conclusion.\n\nEssay:\n"
        )
    elif task == "Social Media Post":
        return (
            f"Write a catchy and engaging social media post about: '{user_input}'.\n"
            "Make it fun, use emojis and hashtags.\n\nPost:\n"
        )
    elif task == "Story":
        return (
            f"Write a short and interesting story about: '{user_input}'.\n\nStory:\n"
        )
    elif task == "Poem":
        return (
            f"Write a beautiful and meaningful poem about: '{user_input}'.\n\nPoem:\n"
        )
    else:  # General Purpose
        return user_input

# UI
st.title("✍️ Multipurpose  Text Generator")
st.markdown("Generate Emails, Essays, Social Media Posts, Stories, and Poems with GPT-2")

# Task selection
task = st.selectbox("Select Generation Type:", [
    "Email", "Essay", "Social Media Post", "Story", "Poem", "General Purpose"
])

# User input
user_input = st.text_area("Enter your topic or idea:", "The importance of time", height=100)
prompt = build_prompt(task, user_input)

# Generation settings
st.subheader("🎛️ Generation Settings")
col1, col2 = st.columns(2)
with col1:
    max_len = st.slider("Max Length", 20, 300, 100)
with col2:
    temperature = st.slider("Temperature", 0.1, 1.5, 1.0)

# Generate
if st.button("🚀 Generate"):
    with st.spinner("Generating..."):
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(
            inputs["input_ids"],
            max_length=max_len,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=1
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        st.success("✨ Generated Text:")
        st.write(generated_text)

        # Download button
        st.download_button(
            label="📥 Download Text",
            data=generated_text,
            file_name=f"{task.lower().replace(' ', '_')}_generated.txt",
            mime="text/plain"
        )

        # Feedback section
        st.markdown("### 🙌 Was this helpful?")
        col_feedback1, col_feedback2 = st.columns(2)
        with col_feedback1:
            if st.button("👍 Yes"):
                st.toast("Thanks for your feedback! 😊")
        with col_feedback2:
            if st.button("👎 No"):
                st.toast("Thanks! We'll keep improving. 💡")

        st.markdown("### Developed by Nawneet Raj")
