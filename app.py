def extract_chart_from_response(response):
    """
    Extract chart specification from model response with improved parsing.
    
    Args:
        response (str): The raw text response from the model
        
    Returns:
        tuple: (cleaned_response, chart_data)
    """
    chart_data = None
    cleaned_response = response
    
    try:
        # Pattern 1: Look for JSON chart spec with improved regex
        chart_pattern = r'\{[\s\n]*["\'"]chart_type["\'][\s\n]*:.*?\}'
        match = re.search(chart_pattern, response, re.DOTALL)
        
        if match:
            chart_spec_str = match.group(0)
            # Clean up the string - replace single quotes with double quotes if needed
            chart_spec_str = chart_spec_str.replace("'", '"')
            
            # Parse the JSON
            chart_data = json.loads(chart_spec_str)
            
            # Remove the chart spec from displayed response
            cleaned_response = re.sub(r'\{[\s\n]*["\'"]chart_type["\'"][\s\n]*:.*?\}', '', response, flags=re.DOTALL)
            
        # Pattern 2: Look for code blocks with JSON chart specifications
        if not chart_data:
            code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
            code_matches = re.findall(code_block_pattern, response, re.DOTALL)
            
            for code_match in code_matches:
                try:
                    # Clean up and parse potential chart JSON
                    potential_chart = json.loads(code_match.replace("'", '"'))
                    if "chart_type" in potential_chart:
                        chart_data = potential_chart
                        # Remove this code block from the response
                        cleaned_response = re.sub(r'```(?:json)?\s*\{[\s\S]*?\}\s*```', '', response, flags=re.DOTALL)
                        break
                except:
                    continue
        
        # Format the chart data if it's in an unexpected format
        if chart_data:
            # Ensure data is properly formatted
            if "data" in chart_data:
                # If data is a list of values but not paired with labels
                if isinstance(chart_data["data"], list) and all(isinstance(x, (int, float)) for x in chart_data["data"]):
                    if "labels" in chart_data and len(chart_data["labels"]) == len(chart_data["data"]):
                        chart_data["data"] = [[label, value] for label, value in zip(chart_data["labels"], chart_data["data"])]
                
                # Handle special case where data is a list of [index, value] pairs
                elif isinstance(chart_data["data"], list) and all(isinstance(x, list) for x in chart_data["data"]):
                    # Add labels if missing
                    if "labels" not in chart_data:
                        if all(len(x) == 2 for x in chart_data["data"]):
                            chart_data["labels"] = [str(item[0]) for item in chart_data["data"]]
    
    except Exception as e:
        print(f"Chart extraction error: {str(e)}")
    
    # Clean up any lingering chart specifications in code blocks
    cleaned_response = re.sub(r'```.*?chart_type.*?```', '', cleaned_response, flags=re.DOTALL)
    
    # Clean up any "Chart Specification" section entirely
    chart_section_pattern = r'(?:\*\*Chart Specification\*\*|\#\#\# Chart Specification).*?(?:\n\n|\Z)'
    cleaned_response = re.sub(chart_section_pattern, '', cleaned_response, flags=re.DOTALL)
    
    return cleaned_response, chart_data


def render_chart(chart_data):
    """
    Renders a chart based on provided chart data specification.
    
    Args:
        chart_data (dict): Chart specification with type, data, labels, etc.
        
    Returns:
        None: Displays chart using streamlit
    """
    try:
        st.subheader(chart_data.get("title", "Data Visualization"))
        
        # Normalize data format
        if isinstance(chart_data["data"][0], list):
            # If data is in [[index, value], [index, value]] format
            values = [item[1] for item in chart_data["data"]]
            # Use provided labels or index from data
            labels = chart_data.get("labels", [str(item[0]) for item in chart_data["data"]])
        else:
            # If data is just a list of values
            values = chart_data["data"]
            labels = chart_data.get("labels", [f"Item {i+1}" for i in range(len(values))])
        
        # Create DataFrame for Altair/Matplotlib
        chart_df = pd.DataFrame({
            'category': labels,
            'value': values
        })
        
        if chart_data["chart_type"].lower() == "bar":
            chart = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X('category', title=chart_data.get("x_axis", "Category"), sort=None),
                y=alt.Y('value', title=chart_data.get("y_axis", "Value"))
            ).properties(
                title=chart_data.get("title", "Bar Chart"),
                width=600,
                height=400
            )
            st.altair_chart(chart, use_container_width=True)
            
        elif chart_data["chart_type"].lower() == "line":
            # Sort by category if it's numeric
            try:
                chart_df['category'] = pd.to_numeric(chart_df['category'])
                chart_df = chart_df.sort_values('category')
            except:
                pass
                
            line_chart = alt.Chart(chart_df).mark_line().encode(
                x=alt.X('category', title=chart_data.get("x_axis", "Category")),
                y=alt.Y('value', title=chart_data.get("y_axis", "Value"))
            ).properties(
                title=chart_data.get("title", "Line Chart"),
                width=600,
                height=400
            )
            st.altair_chart(line_chart, use_container_width=True)
            
        elif chart_data["chart_type"].lower() == "scatter":
            # For scatter plots, ensure we have x and y values
            if isinstance(chart_data["data"][0], list) and len(chart_data["data"][0]) == 2:
                scatter_df = pd.DataFrame(chart_data["data"], columns=['x', 'y'])
                
                scatter_chart = alt.Chart(scatter_df).mark_circle(size=60).encode(
                    x=alt.X('x', title=chart_data.get("x_axis", "X")),
                    y=alt.Y('y', title=chart_data.get("y_axis", "Y"))
                ).properties(
                    title=chart_data.get("title", "Scatter Plot"),
                    width=600,
                    height=400
                )
                st.altair_chart(scatter_chart, use_container_width=True)
            else:
                st.error("Scatter plot requires paired x,y data points")
                
        elif chart_data["chart_type"].lower() == "pie":
            # Create pie chart using matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Ensure values are numeric
            values = [float(v) for v in values]
            
            # Create the pie chart
            ax.pie(values, 
                   labels=labels, 
                   autopct='%1.1f%%',
                   startangle=90,
                   shadow=False)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax.set_title(chart_data.get("title", "Pie Chart"))
            
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error rendering chart: {str(e)}")
        st.write("Chart data received:")
        st.json(chart_data)
