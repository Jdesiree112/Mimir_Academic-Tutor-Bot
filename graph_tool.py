import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import json

def generate_plot(data_json, labels_json, plot_type, title, x_label="", y_label=""):
    """
    Generates a plot (bar, line, or pie) and returns it as an HTML-formatted Base64-encoded image string.
    
    Args:
        data_json (str): JSON string containing data dictionary where keys are labels and values are numerical data.
        labels_json (str): JSON string containing list of labels for the data points.
        plot_type (str): The type of plot to generate ('bar', 'line', or 'pie').
        title (str): The title of the plot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        
    Returns:
        str: An HTML img tag with Base64-encoded plot image.
    """
    try:
        # Parse JSON strings
        data = json.loads(data_json)
        labels = json.loads(labels_json)
    except json.JSONDecodeError as e:
        return f'<p style="color:red;">Error parsing JSON data: {e}</p>'
    
    # Validate inputs
    if not isinstance(data, dict):
        return '<p style="color:red;">Data must be a dictionary with keys as labels and values as numbers.</p>'
    
    if not isinstance(labels, list):
        return '<p style="color:red;">Labels must be a list.</p>'
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract keys and values from the data dictionary
        x_data = list(data.keys())
        y_data = list(data.values())
        
        # Ensure y_data contains numeric values
        try:
            y_data = [float(val) for val in y_data]
        except (ValueError, TypeError):
            return '<p style="color:red;">All data values must be numeric.</p>'
        
        if plot_type == 'bar':
            bars = ax.bar(x_data, y_data)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            # Add value labels on top of bars
            for bar, value in zip(bars, y_data):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value}', ha='center', va='bottom')
                       
        elif plot_type == 'line':
            ax.plot(x_data, y_data, marker='o', linewidth=2, markersize=6)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            
        elif plot_type == 'pie':
            # For pie charts, use labels parameter if provided, otherwise use data keys
            pie_labels = labels if len(labels) == len(y_data) else x_data
            wedges, texts, autotexts = ax.pie(y_data, labels=pie_labels, autopct='%1.1f%%', 
                                            startangle=90, textprops={'fontsize': 10})
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            # Don't set x/y labels for pie charts as they don't make sense
            
        else:
            return f'<p style="color:red;">Invalid plot_type: {plot_type}. Choose "bar", "line", or "pie".</p>'
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Improve layout
        plt.tight_layout()
        
        # Save plot to a BytesIO buffer in memory
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, 
                   facecolor='white', edgecolor='none')
        plt.close(fig)  # Close the plot to free up memory
        
        # Encode the image data to a Base64 string
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Return HTML img tag with proper styling
        return f'''
        <div style="text-align: center; margin: 20px 0;">
            <img src="data:image/png;base64,{img_base64}" 
                 style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);" 
                 alt="{title}" />
        </div>
        '''
        
    except Exception as e:
        plt.close('all')  # Clean up any open figures
        return f'<p style="color:red;">Error generating plot: {str(e)}</p>'