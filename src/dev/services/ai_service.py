"""
Service module for AI-related functionality
"""
import json
import re
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self, api_key, base_url, model):
        """
        Initialize AI service
        
        Args:
            api_key: API key for the AI service
            base_url: Base URL for the API
            model: Model to use for completions
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    def generate_clarifying_questions(self, df_summary, dataset_type):
        """
        Generate clarifying questions about the dataset
        
        Args:
            df_summary: Summary of the dataframe
            dataset_type: Type of dataset
            
        Returns:
            list: List of clarifying questions
        """
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
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            
            if match:
                questions = json.loads(match.group(0))
                return questions[:5] if len(questions) >= 5 else questions + ["Is this data complete?"] * (5 - len(questions))
            else:
                raise ValueError("Could not parse questions from response")
                
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return [
                "Does this dataset contain time-series data?",
                "Are there any missing values that should be handled specially?",
                "Should numerical outliers be excluded from analysis?",
                "Are there specific relationships between columns you're interested in?",
                "Should the analysis focus on trends or current snapshot?"
            ]

    def process_query(self, query, system_prompt, df_summary, chat_history=None):
        """
        Process a natural language query about the data
        
        Args:
            query: User's query
            system_prompt: System prompt for the AI
            df_summary: Summary of the dataframe
            chat_history: Previous chat history
            
        Returns:
            str: AI's response
        """
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
                "visualization_needed": true|false,
                "visualization_type": "bar|line|scatter|pie|none"
            }}
            """
            
            planning_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": planning_prompt}
                ],
                max_completion_tokens=500,
                temperature=0.1
            )
            
            planning_text = planning_response.choices[0].message.content
            match = re.search(r'\{.*\}', planning_text, re.DOTALL)
            
            if not match:
                raise ValueError("Could not parse planning response")
            
            plan = json.loads(match.group(0))
            
            # Second prompt: Execute analysis based on plan
            execution_prompt = f"""
            User query: "{query}"
            
            Analysis plan: {json.dumps(plan)}
            
            Execute this analysis plan on the dataset. 
            
            If visualization is needed, return a specification for it.
            For a chart, provide a JSON specification in this format:
            {{
                "chart_type": "bar|line|scatter|pie",
                "title": "Chart Title",
                "x_axis": "column_name",
                "y_axis": "column_name",
                "data": [[x1, y1], [x2, y2], ...],
                "labels": ["label1", "label2", ...] 
            }}
            
            Return your complete analysis with:
            1. Direct answer to the query
            2. Key numerical findings
            3. Chart specification (if applicable)
            4. 2-3 key insights
            
            Be concise and focused.
            """
            
            messages = [{"role": "system", "content": system_prompt}]
            
            if chat_history:
                for i, exchange in enumerate(chat_history[-3:]):
                    messages.append({"role": "user", "content": exchange["user"]})
                    if "assistant" in exchange:
                        messages.append({"role": "assistant", "content": exchange["assistant"]})
            
            messages.append({"role": "user", "content": execution_prompt})
            
            execution_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=1000,
                temperature=0.2
            )
            
            return execution_response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise 