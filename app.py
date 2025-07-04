import os
import json
import faiss
import numpy as np
import google.generativeai as genai
import re
import streamlit as st
import logging
from typing import List, Dict, Any

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

# Set up Streamlit app layout
st.title("CodeDocs")
st.subheader("An Automated Code Documentation System")

# Initialize the Gemini model with the API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
else:
    st.warning("API Key is missing. Please set the GEMINI_API_KEY environment variable.")

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
  "missing_docstrings": "def add(a, b):\n    '''Adds two numbers and returns the result.'''",
  "potential_improvements": "Add type hints for better clarity."
}
"""

# Function to generate code explanation
def generate_code_explanation(code_snippet: str, output_format="json") -> Dict[str, Any]:
    """Generates a structured explanation of the provided code snippet."""
    try:
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
            You are a code documentation assistant. Analyze the following code and explain it as a story
            in a creative, engaging way. Make the explanation accessible while still being technically accurate.
            Include information about:
            - What the code does
            - The functions and their purpose
            - Any missing documentation
            - Potential improvements

            Code:
            {code_snippet}
            """

        response = model.generate_content(prompt)
        return extract_json_from_response(response.text) if output_format.lower() == "json" else {"story": response.text}
    
    except Exception as e:
        logging.error(f"Error generating explanation: {e}")
        return {"error": f"Error generating explanation: {e}"}

# Function to extract JSON from model response
def extract_json_from_response(text: str) -> Dict[str, Any]:
    """Attempts to extract JSON from the model's response."""
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

# Function to fetch code embeddings
def get_code_embedding(code_snippet: str) -> List[float]:
    """Fetches embedding for the given code snippet."""
    try:
        response = genai.embed_content(model="models/embedding-001", content=code_snippet, task_type="retrieval_document")
        return response['embedding']
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return []

embedding_dimension = 768
embedding_index = faiss.IndexFlatL2(embedding_dimension)
code_snippets = []

# Function to store code snippets in the vector index
def store_code_snippet(code_snippet: str) -> None:
    """Stores the code snippet embedding into the index."""
    embedding = np.array([get_code_embedding(code_snippet)]).astype("float32")
    embedding_index.add(embedding)
    code_snippets.append(code_snippet)

# Function to search for similar code snippets
def search_similar_code(query_snippet: str, top_k=1) -> List[str]:
    """Searches for the most similar code snippets based on embeddings."""
    query_embedding = np.array([get_code_embedding(query_snippet)]).astype("float32")
    D, I = embedding_index.search(query_embedding, top_k)
    return [code_snippets[i] for i in I[0]]

# Function to enhance explanation using RAG (Retrieval-Augmented Generation)
def rag_enhanced_explanation(query_code: str, output_format="json") -> Dict[str, Any]:
    """Generates an enhanced explanation by using similar code snippets as context."""
    similar_snippets = search_similar_code(query_code)
    context = "\n\n".join(similar_snippets)

    if output_format.lower() == "json":
        combined_prompt = f"""
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
        combined_prompt = f"""
        You are a code explanation assistant. Use the context below to help generate better explanation.

        Context:
        {context}

        New Code:
        {query_code}

        Explain this code as an engaging story that a junior developer would find both entertaining and educational.
        Include information about what the code does, its functions, any missing documentation, and potential improvements.
        """

    try:
        response = model.generate_content(combined_prompt)
        return extract_json_from_response(response.text) if output_format.lower() == "json" else {"story": response.text}
    except Exception as e:
        logging.error(f"Error generating RAG-enhanced explanation: {e}")
        return {"error": f"Error generating explanation: {e}"}

# Function to evaluate the explanation quality
def evaluate_explanation_quality(explanation: Dict[str, Any]) -> str:
    """Evaluates the quality of the generated explanation."""
    if "error" in explanation:
        return f"Error in explanation: {explanation['error']}"
    elif "story" in explanation:
        word_count = len(explanation["story"].split())
        if word_count > 100:
            return f"Good story explanation with {word_count} words"
        return f"Story explanation too short: {word_count} words"
    elif "summary" in explanation and len(explanation["summary"].split()) > 3:
        return "Good summary"
    return "Summary is too short or missing"

# Streamlit interface
st.sidebar.title("Settings")
output_format = st.sidebar.selectbox("Choose output format:", ["JSON", "STORY"])
analysis_method = st.sidebar.radio("Use RAG enhancement?", ["Yes", "No"])

st.sidebar.markdown("---")
st.sidebar.subheader("Code Analysis")

code_snippet = st.text_area("Enter or paste your code here:", height=300)

if st.sidebar.button("Analyze Code"):
    with st.spinner("Storing code and generating explanation..."):
        # Store code for RAG enhancement
        try:
            store_code_snippet(code_snippet)
            st.success("Code stored successfully.")
        except Exception as e:
            st.warning(f"Could not store code in vector database. Error: {e}")

        # Generate explanation
        try:
            if analysis_method == "Yes":
                explanation = rag_enhanced_explanation(code_snippet, output_format)
                st.subheader("RAG-Enhanced Explanation")
            else:
                explanation = generate_code_explanation(code_snippet, output_format)
                st.subheader("Basic Explanation")

            if output_format.lower() == "story":
                st.markdown("### 📖 Story Explanation")
                st.write(explanation.get("story", "No story output available."))
            else:
                st.markdown("### 🧾 JSON Explanation")
                st.json(explanation, expanded=True)
            st.info(f"Evaluation: {evaluate_explanation_quality(explanation)}")

        except Exception as e:
            st.error(f"Error generating explanation: {e}")
