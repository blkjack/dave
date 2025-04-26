import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="Data Analyzer",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("📊 Data Analyzer")
st.markdown("""
Upload your CSV file and ask questions about your data in plain English.
This app uses AI to analyze your data and provide insights.
""")

# Sidebar for API configuration
with st.sidebar:
    st.header("API Configuration")
    api_key = st.text_input("Enter your Kluster AI API Key:", type="password")
    model_choice = st.selectbox(
        "Select Model:",
        ["meta-llama/Llama-4-Scout-17B-16E-Instruct", "meta-llama/Llama-3-70B-Instruct"]
    )
    st.divider()
    st.markdown("### About")
    st.markdown("Data Analyzer helps you understand your data through natural language queries.")
    st.markdown("Made with ❤️ using Streamlit and Kluster AI")

# Main area for file upload and data display
uploaded_file = st.file_uploader("Upload your CSV file (max 5000 rows)", type=["csv"])

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None
if 'history' not in st.session_state:
    st.session_state.history = []

# Process uploaded file
if uploaded_file is not None:
    try:
        # Load data with progress bar
        with st.spinner("Loading data..."):
            df = pd.read_csv(uploaded_file, dtype=str, low_memory=True, nrows=5000)
            st.session_state.df = df
            st.session_state.data_summary = df.describe(include='all').to_string()
            st.session_state.history = []
            
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Display basic statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
            
        # Initialize the system prompt in history
        st.session_state.history = [{"role": "system", "content": f"Data Summary:\n{st.session_state.data_summary}"}]
        
    except Exception as e:
        st.error(f"Error loading file: {e}")

# Query section (only show if data is loaded)
if st.session_state.df is not None:
    st.subheader("Ask About Your Data")
    
    # Input for user query
    user_query = st.text_input("Enter your question:", placeholder="E.g., What's the average value in column X?")
    
    if user_query and api_key:
        try:
            with st.spinner("Analyzing data..."):
                # Initialize Kluster AI client
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.kluster.ai/v1"
                )
                
                # Refined query for efficiency
                refined_prompt = f"""
                You are an expert data analyst.
                
                Given the dataset summarized below:
                
                {st.session_state.data_summary}
                
                And the user question: "{user_query}"
                
                Return ONLY:
                - Final calculated table (properly formatted for markdown).
                - Brief key insights (max 4 points).
                - NO Python code, NO explanations, NO long texts.
                
                Keep the answer crisp, clean, and directly useful. Avoid token wastage.
                """
                
                # Query Kluster AI
                completion = client.chat.completions.create(
                    model=model_choice,
                    max_completion_tokens=2000,
                    temperature=0.2,
                    top_p=1,
                    messages=[{"role": "user", "content": refined_prompt}]
                )
                
                response = completion.choices[0].message.content
                
                # Display final answer
                st.subheader("Analysis Results")
                st.markdown(response)
                
                # Add a download button for the results
                result_text = f"Query: {user_query}\n\nResults:\n{response}"
                st.download_button(
                    label="Download Results",
                    data=result_text,
                    file_name="data_analysis_results.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"Error during analysis: {e}")
    elif user_query:
        st.warning("Please enter your API key in the sidebar to analyze data.")

# Display instructions if no file is uploaded
else:
    st.info("👆 Please upload a CSV file to get started.")
    
    # Example queries section
    st.subheader("Example Questions You Can Ask:")
    examples = [
        "What is the distribution of values in column X?",
        "Find the top 5 values in column Y by frequency.",
        "What's the correlation between column A and column B?",
        "Identify any outliers in the numeric columns."
    ]
    
    for example in examples:
        st.markdown(f"- {example}")