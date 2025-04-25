import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


# Load model and tokenizer
@st.cache_resource
def load_model(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model(model_size)

# Title & UI
st.title("✍️ Multipurpose  Text Generator")
st.markdown("Select a content type, customize, and generate amazing AI-powered text!")

# Content type
task = st.selectbox("Select Generation Type:", [
    "Essay", "Social Media Post", "Story", "Poem"
])

# Prompt templates
task_prompts = {
    "Email": "Write a professional email about ",
    "Essay": "Write an essay on ",
    "Social Media Post": "Create a social media post about ",
    "Story": "Write a short story about ",
    "Poem": "Write a poem on ",
    "General Purpose": ""
}

user_input = st.text_area("Enter your topic or idea:", "The importance of time", height=100)
prompt = task_prompts[task] + user_input

# Generation settings
st.subheader("🎛️ Generation Settings")
col1, col2, col3, col4 = st.columns(4)
with col1:
    max_len = st.slider("Max Length", 20, 300, 100)
with col2:
    top_k = st.slider("Top-K", 10, 100, 50)
with col3:
    top_p = st.slider("Top-P", 0.1, 1.0, 0.95)
with col4:
    temperature = st.slider("Temperature", 0.1, 1.5, 1.0)

# Generate button
if st.button("🚀 Generate"):
    with st.spinner("Generating..."):
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(
            inputs["input_ids"],
            max_length=max_len,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
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

        # Feedback
        st.markdown("### 🙌 Was this helpful?")
        col_feedback1, col_feedback2 = st.columns(2)
        with col_feedback1:
            if st.button("👍 Yes"):
                st.toast("Thanks for your feedback! 😊")
        with col_feedback2:
            if st.button("👎 No"):
                st.toast("Thanks! We'll try to improve. 💡")
