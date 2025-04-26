"""
Main application module for the development environment
"""
import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import local modules
from src.dev.utils.data_utils import (
    load_data,
    detect_dataset_type,
    get_data_summary,
    create_test_dataset
)
from src.dev.services.ai_service import AIService
from src.dev.config.settings import (
    API_BASE_URL,
    DEFAULT_MODEL,
    API_KEY,
    MAX_ROWS,
    DEBUG_MODE,
    ENABLE_TEST_DATA,
    SESSION_KEYS,
    DOMAIN_PROMPTS
)

# Page configuration
st.set_page_config(
    page_title="Advanced Data Analyzer (DEV)",
    page_icon="🔧",
    layout="wide"
)

# Initialize session state
for key, default_value in SESSION_KEYS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# App title and description
st.title("🔧 Advanced Data Analyzer (Development)")
st.markdown("""
This is the development version of the Advanced Data Analyzer.
Features and changes are being tested here before being deployed to production.
""")

# Development-specific features
if DEBUG_MODE:
    st.sidebar.header("Development Tools")
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.json({
            "Session State": st.session_state,
            "Config": {k: v for k, v in globals().items() if k.isupper()},
            "Environment": os.environ.get("ENVIRONMENT", "development")
        })

# Sidebar for API configuration
with st.sidebar:
    st.header("API Configuration")
    # Use API key from settings in development
    api_key = API_KEY
    model_choice = st.selectbox(
        "Select Model:",
        ["meta-llama/Llama-4-Scout-17B-16E-Instruct", 
         "meta-llama/Llama-3-70B-Instruct",
         "anthropic.claude-3-opus-20240229"]
    )

    # Credits management section
    st.divider()
    st.header("Credits Management")
    total_credits = 1000
    used_credits = st.session_state.get('used_credits', 0)
    remaining_credits = total_credits - used_credits
    
    st.progress(used_credits/total_credits, f"Credits Used: {used_credits}/{total_credits}")
    
    if remaining_credits < 20:
        st.warning(f"⚠️ Low credits: {remaining_credits} remaining")
    else:
        st.success(f"✅ Credits remaining: {remaining_credits}")
    
    # Reset credits button (for demo purposes)
    if st.button("Reset Credits"):
        st.session_state.used_credits = 0
        st.rerun()

# Add development-specific features
if ENABLE_TEST_DATA:
    st.sidebar.header("Test Data")
    if st.sidebar.button("Load Test Dataset"):
        st.session_state.df = create_test_dataset()
        st.success("Test dataset loaded!")

# Main area for file upload and data display
uploaded_file = st.file_uploader("Upload your CSV file (max 1,000 rows)", type=["csv"])

# Process uploaded file
if uploaded_file is not None:
    try:
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Load data with progress bar
        with st.spinner("Loading data..."):
            st.session_state.df = load_data(uploaded_file, MAX_ROWS, progress_bar)
            st.session_state.data_summary = get_data_summary(st.session_state.df)
            
            # Detect dataset type
            st.session_state.dataset_type = detect_dataset_type(st.session_state.df)
            
            # Initialize understanding score based on dataset type
            if st.session_state.dataset_type != "Unknown":
                st.session_state.understanding_score = 30  # Start with 30% for known dataset types
            
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head(), use_container_width=True)
        
        # Display basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", len(st.session_state.df))
        with col2:
            st.metric("Columns", len(st.session_state.df.columns))
        with col3:
            st.metric("Dataset Type", st.session_state.dataset_type)
        with col4:
            st.metric("Understanding Score", f"{st.session_state.understanding_score}%")
            
        # Set up tabs for Training and Chat
        tab1, tab2 = st.tabs(["Training & Understanding", "Chat & Analysis"])
        
        # Training tab content
        with tab1:
            st.header(f"Training on {st.session_state.dataset_type} Dataset")
            st.markdown(f"""
            The AI has analyzed your dataset and determined it's likely a **{st.session_state.dataset_type}** dataset.
            Current understanding level: **{st.session_state.understanding_score}%**
            
            Please answer these clarifying questions to improve analysis accuracy:
            """)
            
            # Initialize or get clarifying questions
            if not st.session_state.clarifying_questions:
                # Initialize AI service
                ai_service = AIService(api_key, API_BASE_URL, model_choice)
                st.session_state.clarifying_questions = ai_service.generate_clarifying_questions(
                    st.session_state.data_summary,
                    st.session_state.dataset_type
                )
            
            # Display clarifying questions with yes/no toggles
            for i, question in enumerate(st.session_state.clarifying_questions):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**Q{i+1}:** {question}")
                with col2:
                    answer = st.selectbox(
                        f"Answer {i+1}", 
                        options=["Select", "Yes", "No"], 
                        key=f"answer_{i}"
                    )
                    if answer != "Select":
                        st.session_state.clarifying_answers[question] = answer
            
            # Display understanding progress
            if len(st.session_state.clarifying_answers) > 0:
                # Calculate new score: 30% base + 14% per answer (up to 100%)
                new_score = min(100, 30 + len(st.session_state.clarifying_answers) * 14)
                st.session_state.understanding_score = new_score
                st.progress(new_score/100, f"Enhanced understanding: {new_score}%")
            
            # Generate domain-specific system prompt
            if len(st.session_state.clarifying_answers) > 0:
                base_prompt = f"""
                You are an expert data analyst specializing in {st.session_state.dataset_type} data analysis.
                
                Dataset Summary:
                {st.session_state.data_summary}
                
                User clarifications:
                """
                
                # Add clarifying answers
                for q, a in st.session_state.clarifying_answers.items():
                    base_prompt += f"\n- Q: {q}\n  A: {a}"
                
                # Add domain-specific instructions
                if st.session_state.dataset_type in DOMAIN_PROMPTS:
                    base_prompt += DOMAIN_PROMPTS[st.session_state.dataset_type]
                
                base_prompt += """
                \nGeneral instructions:
                - Provide concise, actionable insights
                - When generating charts, use clear labels and titles
                - Present numerical results with appropriate precision
                - Highlight unexpected patterns or anomalies
                - When answering questions, be direct and to the point
                """
                
                st.session_state.system_prompt = base_prompt
                
                st.success("✅ Training complete! Switch to the Chat tab to start analyzing your data.")
            else:
                st.info("Please answer at least one question to improve model understanding.")
        
        # Chat tab content
        with tab2:
            st.header("Chat with Your Data")
            
            # Check if training has been done
            if not st.session_state.get('system_prompt'):
                st.warning("Please complete the training in the first tab before chatting.")
            else:
                # Display chat history
                for chat in st.session_state.chat_history:
                    with st.chat_message("user"):
                        st.write(chat["user"])
                    if "assistant" in chat:
                        with st.chat_message("assistant"):
                            st.write(chat["assistant"])
                
                # Input for new query
                query = st.chat_input("Ask a question about your data...")
                
                if query:
                    # Add user query to chat history
                    st.session_state.chat_history.append({"user": query})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.write(query)
                    
                    # Initialize AI service
                    ai_service = AIService(api_key, API_BASE_URL, model_choice)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing..."):
                            response = ai_service.process_query(
                                query,
                                st.session_state.system_prompt,
                                st.session_state.data_summary,
                                st.session_state.chat_history[:-1]  # Exclude current query
                            )
                            
                            st.write(response)
                    
                    # Update chat history with assistant response
                    st.session_state.chat_history[-1]["assistant"] = response

    except Exception as e:
        st.error(f"Error processing file: {e}")

# Display instructions if no file is uploaded
else:
    st.info("👆 Please upload a CSV file to get started.")
    
    # Example capabilities section
    st.header("App Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Analysis")
        st.markdown("""
        - Upload CSV files (up to 1,000 rows in development)
        - Natural language Q&A with your data
        - Multi-turn conversation memory
        - Automatic ambiguity resolution
        - Domain-adaptive analysis (Finance, Marketing, HR, etc.)
        """)
        
        st.subheader("Example Questions")
        st.markdown("""
        - "What's the average revenue by product category?"
        - "Show me the trend of sales over the last quarter"
        - "Which customers have the highest lifetime value?"
        - "Identify potential anomalies in the transaction data"
        - "Compare performance across different regions"
        """)
        
    with col2:
        st.subheader("Interactive Features")
        st.markdown("""
        - AI training with clarifying questions
        - Domain-specific analysis
        - Automatic chart generation
        - Context retention across queries
        - Query optimization for credit efficiency
        """)
        
        # Sample chart for demonstration
        if ENABLE_TEST_DATA:
            st.subheader("Sample Visualization")
            test_data = create_test_dataset()
            st.bar_chart(test_data, x='date', y='value')
            st.caption("Example of auto-generated charts based on your queries")

# Add footer
st.divider()
st.markdown(f"© {datetime.now().year} Advanced Data Analyzer | Development Version") 