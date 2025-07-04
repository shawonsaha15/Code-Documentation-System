import os
import json
import re
import faiss
import numpy as np
import streamlit as st
#from dotenv import load_dotenv
import google.generativeai as genai

# Load API Key from .env file
#load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

# Few-shot prompt examples
FEW_SHOT_EXAMPLES = """
Examples:

Code:
def add(a, b):
    return a + b

Response:
{
  "summary": "This function performs addition of two numbers.",
  "functions": [
    {
      "name": "add",
      "description": "Returns the sum of two input values a and b."
    }
  ],
  "missing_docstrings": "def add(a, b):\\n    '''Adds two numbers and returns the result.'''",
  "potential_improvements": "Add type hints for better clarity."
}
"""

# Function to generate explanation
def generate_code_explanation(code_snippet: str, output_format="json"):
    if output_format.lower() == "json":
        prompt = f"""
You are a code documentation assistant. Respond only in JSON with:
- summary
- functions (name and description)
- missing docstrings
- potential improvements

{FEW_SHOT_EXAMPLES}

Now analyze this code:

Code:
{code_snippet}
"""
    else:  # Story format
        prompt = f"""
You are a friendly coding assistant helping beginners understand how code works.

Explain the code below step-by-step in simple terms like a walkthrough.

Avoid repeating explanations. Focus on:

- What the code does
- What each function does
- Any missing documentation
- Improvements

Make the explanation clear and concise, as if you're tutoring a beginner.

Code:
{code_snippet}
"""
    response = model.generate_content(prompt)

    if output_format.lower() == "json":
        return extract_json_from_response(response.text)
    else:
        return {"story": response.text.strip()}

# Function to generate tutorial-style explanations
def generate_tutorial_explanation(code_snippet: str):
    # Split code into functions or blocks (for simplicity, we extract function definitions)
    functions = re.findall(r"def (\w+)\((.*?)\):", code_snippet)
    tutorial_steps = []

    # Add a brief overview
    tutorial_steps.append("Let's start learning this code piece by piece. I'll break it down for you.")

    # Guide user through the code
    for func, params in functions:
        step = f"### Step 1: Understanding the Function `{func}`"
        tutorial_steps.append(step)
        tutorial_steps.append(f"Function `{func}` takes the following parameters: `{params}`.")
        tutorial_steps.append(f"This function is likely performing an operation based on these inputs.")
        tutorial_steps.append("Let's explore what the function does next.")

    tutorial_steps.append("Next, I will explain the purpose of any other important sections of the code.")
    tutorial_steps.append("By the end, you'll have a clear understanding of how this code works.")

    # Combine all steps into a final tutorial format
    tutorial = "\n".join(tutorial_steps)
    return {"tutorial": tutorial}

# Helper function to extract valid JSON from model output
def extract_json_from_response(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return {"error": "Model output not valid JSON", "raw_output": text}

# FAISS setup
embedding_dimension = 768
embedding_index = faiss.IndexFlatL2(embedding_dimension)
code_snippets = []

def store_code_snippet(code_snippet: str):
    embedding = np.array([get_code_embedding(code_snippet)]).astype("float32")
    embedding_index.add(embedding)
    code_snippets.append(code_snippet)

def get_code_embedding(code_snippet: str):
    response = genai.embed_content(
        model="models/embedding-001",
        content=code_snippet,
        task_type="retrieval_document"
    )
    return response['embedding']

def search_similar_code(query_snippet: str, top_k=1):
    query_embedding = np.array([get_code_embedding(query_snippet)]).astype("float32")
    D, I = embedding_index.search(query_embedding, top_k)
    return [code_snippets[i] for i in I[0]]

# RAG-enhanced explanation
def rag_enhanced_explanation(query_code: str, output_format="json"):
    similar_snippets = search_similar_code(query_code)
    context = "\n\n".join(similar_snippets)

    if output_format.lower() == "json":
        prompt = f"""
You are a code explanation assistant. Use the context below to help generate better explanation.

Context:
{context}

New Code:
{query_code}

Return your response in this structured JSON format:
{{
  "summary": "...",
  "functions": [...],
  "missing_docstrings": "...",
  "potential_improvements": "..."
}}
"""
    else:
        prompt = f"""
You are a code explanation assistant. Use the context below to help generate better explanation.

Context:
{context}

New Code:
{query_code}

Explain this code as an engaging story that a junior developer would find both entertaining and educational.
Include information about what the code does, its functions, any missing documentation, and potential improvements.
"""
    response = model.generate_content(prompt)
    if output_format.lower() == "json":
        return extract_json_from_response(response.text)
    else:
        return {"story": response.text}

# Streamlit UI
st.set_page_config(page_title="Code Documentation Assistant", layout="centered")
st.title("💬 Code Documentation Assistant")
st.markdown("Generate documentation for your code as **JSON** or **story format** using Gemini and FAISS.")

if "explanation" not in st.session_state:
    st.session_state.explanation = None
if "generated" not in st.session_state:
    st.session_state.generated = False

code_input = st.text_area("📥 Paste your code here", height=200)

col1, col2 = st.columns(2)
with col1:
    mode_choice = st.radio("Choose Mode", ["Explain", "Teach Me"])
with col2:
    if mode_choice != "Teach Me":
        format_choice = st.radio("Choose Output Format", ["json", "story"])
        use_rag = st.radio("Use RAG (retrieve similar code)?", ["No", "Yes"])

if st.button("🚀 Generate Explanation"):
    if not code_input.strip():
        st.warning("Please enter some code first.")
    else:
        try:
            store_code_snippet(code_input)
        except Exception as e:
            st.warning(f"Could not store embedding: {e}")

        if mode_choice == "Teach Me":
            tutorial = generate_tutorial_explanation(code_input)
            st.session_state.explanation = tutorial
            st.subheader("🎓 Teach Me Mode")
        elif use_rag == "Yes":
            explanation = rag_enhanced_explanation(code_input, format_choice)
            st.session_state.explanation = explanation
            st.subheader("🔁 RAG-Enhanced Explanation")
        else:
            explanation = generate_code_explanation(code_input, format_choice)
            st.session_state.explanation = explanation
            st.subheader("📄 Basic Explanation")
        
        st.session_state.generated = True

if st.session_state.generated and st.session_state.explanation:
    explanation = st.session_state.explanation
    if mode_choice == "Teach Me" and "tutorial" in explanation:
        st.markdown("#### Code Tutorial (Step-by-Step)")
        st.markdown(explanation["tutorial"])
    elif format_choice == "json":
        st.json(explanation)
    else:
        if "story" in explanation:
            st.markdown("#### Code Walkthrough (Story Format)")
            st.markdown(explanation["story"])

        else:
            st.error("Failed to generate story format.")
    st.session_state.generated = False
