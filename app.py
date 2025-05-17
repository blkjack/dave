import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import json
import time
import altair as alt
from openai import OpenAI
import re
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Data Analysis and Visualisation Engine",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'understanding_score' not in st.session_state:
    st.session_state.understanding_score = 0
if 'dataset_type' not in st.session_state:
    st.session_state.dataset_type = "Unknown"
if 'clarifying_questions' not in st.session_state:
    st.session_state.clarifying_questions = []
if 'clarifying_answers' not in st.session_state:
    st.session_state.clarifying_answers = {}
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Training"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# App title and description
st.title("📊 Advanced Data Analyzer")
st.markdown("""
Upload your CSV file and interact with your data through natural language. 
This app uses AI to analyze your data, provide insights, and generate visualizations.
""")

# Sidebar for API configuration
with st.sidebar:
    st.header("API Configuration")
    #api_key = st.text_input("Enter your Kluster AI API Key:", type="password")
    api_key = '280348c1-eddc-4b9d-aba6-e7d2641c968d'
    model_choice = st.selectbox(
        "Select Model:",
        ["meta-llama/Llama-4-Scout-17B-16E-Instruct", 
         "meta-llama/Llama-3-70B-Instruct",
         "anthropic.claude-3-opus-20240229"]
    )

    # Credits management section
    st.divider()
    st.header("Credits Management")
    # In a real app, you might track actual API usage
    total_credits = 100
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
    
    st.divider()
    st.markdown("### About")
    st.markdown("Agent DAVE! your data analysis and Visualization engine helps you understand your data through natural language.")
    st.markdown("Made with 🚀")

# Function to detect dataset type
def detect_dataset_type(df):
    # Get column names and check for keywords
    cols = [col.lower() for col in df.columns]
    col_string = " ".join(cols)
    
    # Simple rule-based detection
    if any(term in col_string for term in ['price', 'cost', 'revenue', 'profit', 'expense', 'budget', 'sales']):
        return "Finance"
    elif any(term in col_string for term in ['campaign', 'customer', 'click', 'conversion', 'ctr', 'roi', 'lead']):
        return "Marketing"
    elif any(term in col_string for term in ['employee', 'salary', 'hire', 'performance', 'department', 'manager']):
        return "HR"
    elif any(term in col_string for term in ['patient', 'diagnosis', 'treatment', 'doctor', 'hospital', 'medication']):
        return "Healthcare"
    elif any(term in col_string for term in ['student', 'grade', 'course', 'class', 'teacher', 'school']):
        return "Education"
    else:
        return "General"

# Function to simulate calculating model understanding
def calculate_understanding(df):
    # In a real app, this might involve actually testing the model
    # Here we just simulate a score
    time.sleep(1)  # Simulate processing
    # Return a score between 70 and 95
    return round(70 + 25 * np.random.random(), 1)

# Function to generate clarifying questions
def generate_clarifying_questions(client, df_summary, dataset_type):
    if not client:
        # Return placeholder questions if no API client
        return [
            "Does this dataset include time-series data?",
            "Are there any missing values that should be handled specially?",
            "Should numerical outliers be excluded from analysis?",
            "Are there specific relationships between columns you're interested in?",
            "Should the analysis focus on trends or current snapshot?"
        ]
    
    try:
        prompt = f"""
        You are an expert data scientist. You've been given a new dataset with the following summary:
        
        {df_summary}
        
        Dataset type: {dataset_type}
        
        Please generate EXACTLY 5 yes/no questions that would help you better understand this dataset 
        before analyzing it. These questions should help clarify ambiguities and improve analysis accuracy.
        
        Return ONLY the questions in a JSON array format, like this:
        ["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?"]
        
        Each question MUST be answerable with a simple yes or no.
        """
        
        # Use minimal tokens to save credits
        response = client.chat.completions.create(
            model=model_choice,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=500,
            temperature=0.1
        )
        
        # Extract JSON from response
        response_text = response.choices[0].message.content
        match = re.search(r'\[.*\]', response_text, re.DOTALL)
        
        if match:
            questions = json.loads(match.group(0))
            # Ensure we have exactly 5 questions
            return questions[:5] if len(questions) >= 5 else questions + ["Is this data complete?"] * (5 - len(questions))
        else:
            raise ValueError("Could not parse questions from response")
            
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return [
            "Does this dataset contain time-series data?",
            "Are there any missing values that should be handled specially?",
            "Should numerical outliers be excluded from analysis?",
            "Are there specific relationships between columns you're interested in?",
            "Should the analysis focus on trends or current snapshot?"
        ]

# Function to generate domain-specific system prompt
def get_domain_prompt(dataset_type, df_summary, clarifying_answers):
    # Base system prompt
    base_prompt = f"""
    You are an expert data analyst specializing in {dataset_type} data analysis.
    
    Dataset Summary:
    {df_summary}
    
    User clarifications:
    """
    
    # Add clarifying answers
    for q, a in clarifying_answers.items():
        base_prompt += f"\n- Q: {q}\n  A: {a}"
    
    # Add domain-specific instructions
    if dataset_type == "Finance":
        base_prompt += """
        \nSpecial instructions for Finance data:
        - When analyzing financial metrics, consider YoY growth and margins
        - Express monetary values with appropriate currency symbols
        - Focus on ROI, profitability, and financial performance metrics
        - Consider seasonality in financial data
        """
    elif dataset_type == "Marketing":
        base_prompt += """
        \nSpecial instructions for Marketing data:
        - Focus on conversion rates, CAC, CLV, and ROI metrics
        - Consider campaign performance and audience segmentation
        - Look for correlations between marketing efforts and outcomes
        - Provide actionable marketing insights
        """
    elif dataset_type == "HR":
        base_prompt += """
        \nSpecial instructions for HR data:
        - Focus on employee retention, satisfaction, and performance metrics
        - Consider departmental differences and team dynamics
        - Analyze compensation equity and promotion patterns
        - Look for factors that affect recruitment and turnover
        """
    elif dataset_type == "Healthcare":
        base_prompt += """
        \nSpecial instructions for Healthcare data:
        - Focus on patient outcomes, treatment efficacy, and care quality
        - Consider demographic factors in health outcomes
        - Analyze resource utilization and operational efficiency
        - Be mindful of privacy considerations in analysis
        """
    elif dataset_type == "Education":
        base_prompt += """
        \nSpecial instructions for Education data:
        - Focus on student performance, learning outcomes, and engagement
        - Consider demographic factors in educational outcomes
        - Analyze teaching effectiveness and resource allocation
        - Look for patterns in student achievement and growth
        """
    
    base_prompt += """
    \nGeneral instructions:
    - Provide concise, actionable insights
    - When generating charts, use clear labels and titles
    - Present numerical results with appropriate precision
    - Highlight unexpected patterns or anomalies
    - When answering questions, be direct and to the point
    """
    
    return base_prompt

# Function to process user query with chain of prompts
def process_query(client, query, system_prompt, df_summary, chat_history=None):
    if not client:
        return "Please enter your API key in the sidebar to analyze data."
    
    try:
        # First prompt: Query understanding and planning
        planning_prompt = f"""
        User query: "{query}"
        
        Dataset summary:
        {df_summary}
        
        First, identify what the user is asking for. Then create a step-by-step plan to answer their query.
        Return ONLY the plan in JSON format like this:
        {{
            "query_type": "statistical_analysis|visualization|prediction|simple_lookup|complex_analysis",
            "columns_needed": ["col1", "col2"],
            "analysis_steps": ["step1", "step2", "step3"],
            "analysis_description": "Brief description of what the analysis will show"
        }}
        """
        
        # Run the planning prompt
        planning_response = client.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": planning_prompt}
            ],
            max_completion_tokens=500,
            temperature=0.1
        )
        
        planning_text = planning_response.choices[0].message.content
        
        # Extract JSON from response
        match = re.search(r'\{.*\}', planning_text, re.DOTALL)
        if not match:
            raise ValueError("Could not parse planning response")
        
        plan = json.loads(match.group(0))
        
        # Second prompt: Execute analysis based on plan
        execution_prompt = f"""
        User query: "{query}"
        
        Analysis plan: {json.dumps(plan)}
        
        Execute this analysis plan on the dataset. 
        
        Return your analysis in this EXACT format:
        
        ### Direct Answer
        [One sentence answer to the query]
        
        ### Key Numerical Findings
        - [Finding 1]
        - [Finding 2]
        - [Finding 3]
        
        ### Key Insights
        - [Insight 1]
        - [Insight 2]
        
        ### Next Action
        [Recommended next action]
        
        ### Reason
        [Brief explanation of why this action is recommended]
        
        Be extremely concise. Use bullet points. No additional text or formatting.
        """
        
        # Messages for execution, including chat history if provided
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history if provided (limited to last 3 exchanges to save tokens)
        if chat_history:
            for i, exchange in enumerate(chat_history[-3:]):
                messages.append({"role": "user", "content": exchange["user"]})
                if "assistant" in exchange:
                    messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        messages.append({"role": "user", "content": execution_prompt})
        
        # Run the execution prompt
        execution_response = client.chat.completions.create(
            model=model_choice,
            messages=messages,
            max_completion_tokens=1000,
            temperature=0.2
        )
        
        # Increment used credits
        st.session_state.used_credits = st.session_state.get('used_credits', 0) + 1
        
        return execution_response.choices[0].message.content
        
    except Exception as e:
        return f"Error processing query: {e}"

# Main area for file upload and data display
uploaded_file = st.file_uploader("Upload your CSV file (max 30,000 rows)", type=["csv"])

# Process uploaded file
if uploaded_file is not None:
    try:
        # Load data with progress bar
        with st.spinner("Loading data..."):
            df = pd.read_csv(uploaded_file, dtype=str, low_memory=True, nrows=30000)
            st.session_state.df = df
            
            # Convert numeric columns from string to float for analysis
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
                    
            st.session_state.data_summary = df.describe(include='all').to_string()
            
            # Detect dataset type
            st.session_state.dataset_type = detect_dataset_type(df)
            
            # Calculate understanding score
            st.session_state.understanding_score = calculate_understanding(df)
            
            # Reset session state for a new file
            st.session_state.history = []
            st.session_state.clarifying_questions = []
            st.session_state.clarifying_answers = {}
            st.session_state.chat_history = []
            
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Display basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Dataset Type", st.session_state.dataset_type)
        with col4:
            st.metric("Understanding Score", f"{st.session_state.understanding_score}%")
            
        # Initialize the system prompt in history
        st.session_state.history = [{"role": "system", "content": f"Data Summary:\n{st.session_state.data_summary}"}]
        
        # Set up tabs for Training, Q&A, and Charts
        tab1, tab2, tab3 = st.tabs(["Training & Understanding", "Q&A", "Charts"])
        
        # Training tab content
        with tab1:
            st.header(f"Training on {st.session_state.dataset_type} Dataset")
            st.markdown(f"""
            The AI has analyzed your dataset and determined it's likely a **{st.session_state.dataset_type}** dataset.
            Current understanding level: **{st.session_state.understanding_score}%**
            
            Please answer these clarifying questions to improve analysis accuracy:
            """)
            
            # Initialize or get clarifying questions
            if not st.session_state.clarifying_questions and api_key:
                # Initialize Kluster AI client
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.kluster.ai/v1"
                )
                st.session_state.clarifying_questions = generate_clarifying_questions(
                    client, 
                    st.session_state.data_summary,
                    st.session_state.dataset_type
                )
            elif not st.session_state.clarifying_questions:
                st.session_state.clarifying_questions = [
                    "Does this dataset include time-series data?",
                    "Are there any missing values that should be handled specially?",
                    "Should numerical outliers be excluded from analysis?",
                    "Are there specific relationships between columns you're interested in?",
                    "Should the analysis focus on trends or current snapshot?"
                ]
            
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
                new_score = min(99, st.session_state.understanding_score + len(st.session_state.clarifying_answers) * 2)
                st.progress(new_score/100, f"Enhanced understanding: {new_score}%")
            
            # Generate domain-specific system prompt
            if len(st.session_state.clarifying_answers) > 0:
                system_prompt = get_domain_prompt(
                    st.session_state.dataset_type,
                    st.session_state.data_summary,
                    st.session_state.clarifying_answers
                )
                st.session_state.system_prompt = system_prompt
                
                st.success("✅ Training complete! Switch to the Q&A or Charts tab to start analyzing your data.")
            else:
                st.info("Please answer at least one question to improve model understanding.")
        
        # Q&A tab content
        with tab2:
            st.header("Ask Questions About Your Data")
            
            # Check if training has been done
            if not st.session_state.get('system_prompt'):
                st.warning("Please complete the training in the first tab before asking questions.")
            else:
                # Display chat history
                for chat in st.session_state.chat_history:
                    with st.chat_message("user"):
                        st.write(chat["user"])
                    if "assistant" in chat:
                        with st.chat_message("assistant"):
                            # Display the response in markdown format
                            st.markdown(chat["assistant"])
                
                # Input for new query
                query = st.chat_input("Ask a question about your data...")
                
                if query:
                    # Add user query to chat history
                    st.session_state.chat_history.append({"user": query})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.write(query)
                    
                    # Initialize Kluster AI client
                    if api_key:
                        client = OpenAI(
                            api_key=api_key,
                            base_url="https://api.kluster.ai/v1"
                        )
                        
                        with st.chat_message("assistant"):
                            with st.spinner("Analyzing..."):
                                try:
                                    # Process the query
                                    response = process_query(
                                        client,
                                        query,
                                        st.session_state.system_prompt,
                                        st.session_state.data_summary,
                                        st.session_state.chat_history[:-1]  # Exclude current query
                                    )
                                    
                                    # Display the formatted response
                                    st.markdown(response)
                                    
                                    # Update chat history with assistant response
                                    st.session_state.chat_history[-1]["assistant"] = response
                                    
                                except Exception as e:
                                    error_message = f"""
                                    ### Error Processing Query
                                    
                                    An error occurred while processing your query:
                                    ```
                                    {str(e)}
                                    ```
                                    
                                    Please try rephrasing your question or check if the data contains the information you're looking for.
                                    """
                                    st.error(error_message)
                                    st.session_state.chat_history[-1]["assistant"] = error_message
                    else:
                        with st.chat_message("assistant"):
                            st.error("Please enter your API key in the sidebar to chat with your data.")
        
        # Charts tab content
        with tab3:
            st.header("Data Visualization")
            
            # Check if training has been done
            if not st.session_state.get('system_prompt'):
                st.warning("Please complete the training in the first tab before creating charts.")
            else:
                # Chart type selection
                chart_type = st.selectbox(
                    "Select Chart Type",
                    ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart"]
                )
                
                # Column selection
                if st.session_state.df is not None:
                    numeric_columns = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns
                    categorical_columns = st.session_state.df.select_dtypes(include=['object']).columns
                    
                    if chart_type in ["Bar Chart", "Pie Chart"]:
                        x_column = st.selectbox("Select Category Column", categorical_columns)
                        y_column = st.selectbox("Select Value Column", numeric_columns)
                        
                        if st.button("Generate Chart"):
                            try:
                                # Create the chart
                                if chart_type == "Bar Chart":
                                    chart_data = st.session_state.df.groupby(x_column)[y_column].count().reset_index()
                                    st.bar_chart(chart_data, x=x_column, y=y_column, use_container_width=True)
                                else:  # Pie Chart
                                    chart_data = st.session_state.df.groupby(x_column)[y_column].count()
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.pie(chart_data, labels=chart_data.index, autopct='%1.1f%%')
                                    ax.set_title(f"Distribution of {y_column} by {x_column}")
                                    st.pyplot(fig)
                                    
                                # Display insights
                                st.subheader("Key Insights")
                                st.markdown(f"""
                                - The chart shows the distribution of {y_column} across different {x_column} categories
                                - The highest value is {chart_data[y_column].max()} for {chart_data.loc[chart_data[y_column].idxmax(), x_column]}
                                - The lowest value is {chart_data[y_column].min()} for {chart_data.loc[chart_data[y_column].idxmin(), x_column]}
                                """)
                            except Exception as e:
                                st.error(f"Error generating chart: {str(e)}")
                    
                    elif chart_type == "Line Chart":
                        x_column = st.selectbox("Select X-axis Column", numeric_columns)
                        y_column = st.selectbox("Select Y-axis Column", numeric_columns)
                        
                        if st.button("Generate Chart"):
                            try:
                                chart_data = st.session_state.df.sort_values(x_column)
                                st.line_chart(chart_data, x=x_column, y=y_column, use_container_width=True)
                                
                                # Display insights
                                st.subheader("Key Insights")
                                st.markdown(f"""
                                - The line chart shows the trend of {y_column} over {x_column}
                                - The highest value is {chart_data[y_column].max()}
                                - The lowest value is {chart_data[y_column].min()}
                                """)
                            except Exception as e:
                                st.error(f"Error generating chart: {str(e)}")
                    
                    elif chart_type == "Scatter Plot":
                        x_column = st.selectbox("Select X-axis Column", numeric_columns)
                        y_column = st.selectbox("Select Y-axis Column", numeric_columns)
                        color_column = st.selectbox("Select Color Column (Optional)", ["None"] + list(categorical_columns))
                        
                        if st.button("Generate Chart"):
                            try:
                                if color_column == "None":
                                    st.scatter_chart(st.session_state.df, x=x_column, y=y_column, use_container_width=True)
                                else:
                                    st.scatter_chart(st.session_state.df, x=x_column, y=y_column, color=color_column, use_container_width=True)
                                
                                # Display insights
                                st.subheader("Key Insights")
                                st.markdown(f"""
                                - The scatter plot shows the relationship between {x_column} and {y_column}
                                - The correlation coefficient is {st.session_state.df[x_column].corr(st.session_state.df[y_column]):.2f}
                                """)
                            except Exception as e:
                                st.error(f"Error generating chart: {str(e)}")

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
        - Upload CSV files (up to 30,000 rows)
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
        data = pd.DataFrame({
            'Category': ['A', 'B', 'C', 'D', 'E'],
            'Value': [5, 7, 3, 9, 6]
        })
        
        st.subheader("Sample Visualization")
        st.bar_chart(data, x='Category', y='Value')
        st.caption("Example of auto-generated charts based on your queries")

# Add footer
st.divider()
st.markdown(f"© {datetime.now().year} Advanced Data Analyzer | Last updated: April 2025")
