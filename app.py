import os
import json
import faiss
import numpy as np
import google.generativeai as genai
import re
import streamlit as st

# Set up Streamlit app layout
st.title("CodeDocs - An Automated Code Documentation System")

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

    if output_format.lower() == "json":
        return extract_json_from_response(response.text)
    else:
        return {"story": response.text}

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

def get_code_embedding(code_snippet: str):
    response = genai.embed_content(
        model="models/embedding-001",
        content=code_snippet,
        task_type="retrieval_document"
    )
    return response['embedding']

embedding_dimension = 768
embedding_index = faiss.IndexFlatL2(embedding_dimension)
code_snippets = []

def store_code_snippet(code_snippet: str):
    embedding = np.array([get_code_embedding(code_snippet)]).astype("float32")
    embedding_index.add(embedding)
    code_snippets.append(code_snippet)

def search_similar_code(query_snippet: str, top_k=1):
    query_embedding = np.array([get_code_embedding(query_snippet)]).astype("float32")
    D, I = embedding_index.search(query_embedding, top_k)
    return [code_snippets[i] for i in I[0]]

def rag_enhanced_explanation(query_code: str, output_format="json"):
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

    response = model.generate_content(combined_prompt)

    if output_format.lower() == "json":
        return extract_json_from_response(response.text)
    else:
        return {"story": response.text}

def evaluate_explanation_quality(explanation: dict):
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

def print_explanation(explanation, output_format):
    if output_format.lower() == "json":
        st.json(explanation, expanded=True)
    else:
        if "story" in explanation:
            st.write("\n--- CODE STORY ---\n")
            st.write(explanation["story"])
            st.write("\n-----------------\n")
        else:
            st.write("Error generating story format")
            st.write(explanation)

# Streamlit interface
code_snippet = st.text_area("Enter or paste your code here:", height=300)

if code_snippet:
    output_format = st.selectbox("Choose output format:", ["JSON", "STORY"])

    analysis_method = st.radio("Use RAG enhancement?", ["Yes", "No"])

    if st.button("Analyze Code"):
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

            print_explanation(explanation, output_format)
            st.info(f"Evaluation: {evaluate_explanation_quality(explanation)}")

        except Exception as e:
            st.error(f"Error generating explanation: {e}")
