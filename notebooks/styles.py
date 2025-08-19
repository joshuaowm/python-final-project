"""
Satellite Image Classification Notebook - UI Styles
====================================================
This module contains all the styling functions and HTML templates for the satellite 
image classification notebook to keep the main notebook clean and focused on logic.
"""

from IPython.display import HTML, display
from datetime import datetime
import os


class NotebookStyles:
    """Container for all notebook styling and UI components"""
    
    # Color schemes for different contexts
    COLORS = {
        'primary': {'bg': '#667eea', 'text': '#ffffff'},
        'success': {'bg': '#d4edda', 'border': '#c3e6cb', 'text': '#155724'},
        'warning': {'bg': '#fff3cd', 'border': '#ffeaa7', 'text': '#856404'},
        'danger': {'bg': '#f8d7da', 'border': '#f5c6cb', 'text': '#721c24'},
        'info': {'bg': '#cce5ff', 'border': '#80bdff', 'text': '#004085'},
        'secondary': {'bg': '#f8f9fa', 'border': '#dee2e6', 'text': '#495057'},
    }
    
    @staticmethod
    def session_header(title="🛰️ Satellite Image Classification Demo", subtitle=""):
        """Create the main session header"""
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        return HTML(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   padding: 20px; border-radius: 15px; color: white; margin: 20px 0;">
            <h2 style="margin: 0; font-weight: 300;">{title}</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1em;">
                {subtitle}
            </p>
            <div style="margin-top: 10px; font-size: 0.9em; opacity: 0.8;">
                🔬 Analysis Session Started: {timestamp}
            </div>
        </div>
        """)
        
    @staticmethod
    def session_header_2(title="🛰️ Semantic Segmentation Demo", subtitle=""):
        """Create the main session header"""
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        return HTML(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   padding: 20px; border-radius: 15px; color: white; margin: 20px 0;">
            <h2 style="margin: 0; font-weight: 300;">{title}</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1em;">
                {subtitle}
            </p>
            <div style="margin-top: 10px; font-size: 0.9em; opacity: 0.8;">
                🔬 Analysis Session Started: {timestamp}
            </div>
        </div>
        """)
    
    @staticmethod
    def loading_spinner(text="Loading..."):
        """Create a loading spinner with custom text"""
        return HTML(f"""
        <div style="text-align: center; padding: 20px;">
            <div style="border: 4px solid #f3f3f3; border-top: 4px solid #3498db; 
                       border-radius: 50%; width: 40px; height: 40px; 
                       animation: spin 1s linear infinite; margin: 0 auto;">
            </div>
            <p style="margin-top: 10px; color: #666;">{text}</p>
        </div>
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """)
    
    @staticmethod
    def success_card(title, message, stats=None):
        """Create a success notification card"""
        stats_html = ""
        if stats:
            stats_items = []
            for key, value in stats.items():
                stats_items.append(f"""
                <div style="text-align: center;">
                    <h3 style="margin: 0; font-size: 2em;">{value}</h3>
                    <p style="margin: 5px 0 0 0;">{key}</p>
                </div>
                """)
            stats_html = f"""
            <div style="display: flex; justify-content: space-around; margin-top: 15px;">
                {''.join(stats_items)}
            </div>
            """
        
        return HTML(f"""
        <div style="background: {NotebookStyles.COLORS['success']['bg']}; 
                   border: 1px solid {NotebookStyles.COLORS['success']['border']}; 
                   border-radius: 10px; padding: 15px; margin: 10px 0;">
            <h4 style="color: {NotebookStyles.COLORS['success']['text']}; margin: 0 0 10px 0;">
                ✅ {title}
            </h4>
            <p style="color: {NotebookStyles.COLORS['success']['text']}; margin: 0;">
                {message}
            </p>
            {stats_html}
        </div>
        """)
    
    @staticmethod
    def error_card(title, message):
        """Create an error notification card"""
        return HTML(f"""
        <div style="background: {NotebookStyles.COLORS['danger']['bg']}; 
                   border: 1px solid {NotebookStyles.COLORS['danger']['border']}; 
                   border-radius: 10px; padding: 15px; margin: 10px 0;">
            <h4 style="color: {NotebookStyles.COLORS['danger']['text']}; margin: 0 0 10px 0;">
                ❌ {title}
            </h4>
            <p style="color: {NotebookStyles.COLORS['danger']['text']}; margin: 0;">
                {message}
            </p>
        </div>
        """)
    
    @staticmethod
    def warning_card(title, message):
        """Create a warning notification card"""
        return HTML(f"""
        <div style="background: {NotebookStyles.COLORS['warning']['bg']}; 
                   border: 1px solid {NotebookStyles.COLORS['warning']['border']}; 
                   border-radius: 10px; padding: 15px; margin: 10px 0;">
            <h4 style="color: {NotebookStyles.COLORS['warning']['text']}; margin: 0 0 10px 0;">
                ⚠️ {title}
            </h4>
            <p style="color: {NotebookStyles.COLORS['warning']['text']}; margin: 0;">
                {message}
            </p>
        </div>
        """)
    
    @staticmethod
    def info_card(title, message):
        """Create an info notification card"""
        return HTML(f"""
        <div style="background: {NotebookStyles.COLORS['info']['bg']}; 
                   border: 1px solid {NotebookStyles.COLORS['info']['border']}; 
                   border-radius: 10px; padding: 15px; margin: 10px 0;">
            <h4 style="color: {NotebookStyles.COLORS['info']['text']}; margin: 0 0 10px 0;">
                📈 {title}
            </h4>
            <p style="color: {NotebookStyles.COLORS['info']['text']}; margin: 0;">
                {message}
            </p>
        </div>
        """)
    
    @staticmethod
    def section_header(title, subtitle="", icon="🔧"):
        """Create a section header"""
        subtitle_html = f"<p style='margin: 5px 0 0 0; opacity: 0.8;'>{subtitle}</p>" if subtitle else ""
        return HTML(f"""
        <div style="background: {NotebookStyles.COLORS['secondary']['bg']}; 
                   border-left: 4px solid #007bff; padding: 20px; margin: 20px 0; 
                   border-radius: 0 10px 10px 0;">
            <h3 style="color: {NotebookStyles.COLORS['secondary']['text']}; margin: 0;">
                {icon} {title}
            </h3>
            {subtitle_html}
        </div>
        """)
    
    @staticmethod
    def image_preview(image_path, title="Image Preview"):
        """Create a styled image preview"""
        if os.path.exists(image_path):
            return HTML(f"""
            <div style="text-align: center; margin: 20px 0; 
                       padding: 20px; background: #f8f9fa; border-radius: 15px;">
                <h4 style="color: #495057; margin-bottom: 15px;">
                    📸 {title}
                </h4>
                <div style="display: inline-block; border: 3px solid #dee2e6; 
                           border-radius: 10px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                    <img src="{image_path}" style="max-width: 400px; max-height: 300px; display: block;">
                </div>
                <p style="margin-top: 10px; color: #6c757d; font-style: italic;">
                    {os.path.basename(image_path)}
                </p>
            </div>
            """)
        else:
            return NotebookStyles.warning_card(
                "Image Not Found",
                f"Could not locate: {image_path}"
            )
    
    @staticmethod
    def progress_bar(width=0, text="Initializing..."):
        """Create a progress bar"""
        return HTML(f"""
        <div id="progress-container" style="margin: 20px 0;">
            <div style="background: #e9ecef; height: 20px; border-radius: 10px; overflow: hidden;">
                <div id="progress-bar" style="background: linear-gradient(90deg, #28a745, #20c997); 
                                            height: 100%; width: {width}%; transition: width 0.3s ease;"></div>
            </div>
            <p id="progress-text" style="text-align: center; margin-top: 10px; color: #6c757d;">
                {text}
            </p>
        </div>
        """)
    
    @staticmethod
    def model_summary_card(models):
        """Create a model summary card"""
        model_list = [f"<li style='margin: 5px 0;'>🔹 {name}</li>" for name in models.keys()]
        return HTML(f"""
        <div style="background: {NotebookStyles.COLORS['secondary']['bg']}; 
                   border-left: 4px solid #007bff; padding: 15px; margin: 10px 0;">
            <h5 style="color: {NotebookStyles.COLORS['secondary']['text']}; margin: 0 0 10px 0;">
                Available Models:
            </h5>
            <ul style="margin: 0; padding-left: 20px; color: #6c757d;">
                {''.join(model_list)}
            </ul>
        </div>
        """)
    
    @staticmethod
    def classification_summary(results):
        """Create a classification results summary"""
        successful_models = sum(1 for result in results.values() if 'error' not in result)
        failed_models = len(results) - successful_models
        
        return HTML(f"""
        <div style="background: linear-gradient(135deg, #28a745, #20c997); 
                   color: white; padding: 20px; border-radius: 15px; margin: 20px 0;">
            <h4 style="margin: 0 0 15px 0;">🎯 Classification Summary</h4>
            <div style="display: flex; justify-content: space-around; text-align: center;">
                <div>
                    <h2 style="margin: 0; font-size: 2em;">{successful_models}</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Successful</p>
                </div>
                <div>
                    <h2 style="margin: 0; font-size: 2em;">{failed_models}</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Failed</p>
                </div>
                <div>
                    <h2 style="margin: 0; font-size: 2em;">{len(results)}</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Total Models</p>
                </div>
            </div>
        </div>
        """)
    
    @staticmethod
    def quick_results_preview(results):
        """Create a quick preview of classification results"""
        html_content = """
        <div style="background: #f8f9fa; border-radius: 15px; padding: 20px; margin: 20px 0;">
            <h4 style="color: #495057; margin: 0 0 15px 0;">⚡ Quick Results Preview</h4>
        """
        
        for model_name, result in results.items():
            if 'error' not in result:
                top_prediction = result['predictions'][0]
                confidence = top_prediction['confidence']
                
                # Determine confidence styling
                if confidence > 0.8:
                    emoji, color = "🎯", "#28a745"
                elif confidence > 0.5:
                    emoji, color = "🎲", "#ffc107"
                else:
                    emoji, color = "❓", "#dc3545"
                
                html_content += f"""
                <div style="background: white; margin: 8px 0; padding: 12px; 
                           border-radius: 8px; border-left: 4px solid {color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: bold; color: #495057;">
                            {emoji} {model_name}
                        </span>
                        <span style="color: {color}; font-weight: bold;">
                            {confidence:.3f}
                        </span>
                    </div>
                    <div style="color: #6c757d; margin-top: 5px;">
                        {top_prediction['class']}
                    </div>
                </div>
                """
            else:
                html_content += f"""
                <div style="background: white; margin: 8px 0; padding: 12px; 
                           border-radius: 8px; border-left: 4px solid #dc3545;">
                    <span style="color: #dc3545;">❌ {model_name}: {result['error']}</span>
                </div>
                """
        
        html_content += "</div>"
        return HTML(html_content)
    
    @staticmethod
    def interpretation_guide():
        """Create a guide for interpreting results"""
        return HTML("""
        <div style="background: #fff; border: 2px solid #dee2e6; 
                   border-radius: 15px; padding: 20px; margin: 20px 0;">
            <h4 style="color: #495057; margin: 0 0 15px 0;">
                📖 How to Interpret the Results
            </h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h5 style="color: #28a745; margin: 0 0 10px 0;">✅ High Confidence (>0.8)</h5>
                    <p style="margin: 0; color: #6c757d;">The model is very confident in its prediction</p>
                </div>
                <div>
                    <h5 style="color: #ffc107; margin: 0 0 10px 0;">⚠️ Medium Confidence (0.5-0.8)</h5>
                    <p style="margin: 0; color: #6c757d;">The model has moderate confidence</p>
                </div>
                <div>
                    <h5 style="color: #dc3545; margin: 0 0 10px 0;">❓ Low Confidence (<0.5)</h5>
                    <p style="margin: 0; color: #6c757d;">The model is uncertain about its prediction</p>
                </div>
                <div>
                    <h5 style="color: #6c757d; margin: 0 0 10px 0;">❌ Error</h5>
                    <p style="margin: 0; color: #6c757d;">The model failed to process the image</p>
                </div>
            </div>
        </div>
        """)
    
    @staticmethod
    def detailed_results_display(results, image_name):
        """Create an enhanced HTML display of detailed results"""
        
        html_content = f"""
        <div style="background: #f8f9fa; padding: 20px; border-radius: 15px; margin: 20px 0;">
            <h2 style="color: #495057; margin: 0 0 20px 0; text-align: center;">
                🔬 Detailed Analysis Report
            </h2>
            <div style="text-align: center; margin-bottom: 20px; color: #6c757d;">
                <strong>Image:</strong> {os.path.basename(image_name)}
            </div>
        """
        
        for model_name, result in results.items():
            if 'error' in result:
                html_content += f"""
                <div style="background: #f8d7da; border-left: 5px solid #dc3545; 
                           padding: 15px; margin: 10px 0; border-radius: 0 10px 10px 0;">
                    <h4 style="color: #721c24; margin: 0 0 10px 0;">
                        ❌ {model_name}
                    </h4>
                    <p style="color: #721c24; margin: 0;">
                        Error: {result['error']}
                    </p>
                </div>
                """
            else:
                # Determine confidence color scheme
                top_confidence = result['predictions'][0]['confidence']
                if top_confidence > 0.8:
                    color_scheme = {"bg": "#d4edda", "border": "#28a745", "text": "#155724", "emoji": "🎯"}
                elif top_confidence > 0.5:
                    color_scheme = {"bg": "#fff3cd", "border": "#ffc107", "text": "#856404", "emoji": "⚠️"}
                else:
                    color_scheme = {"bg": "#f8d7da", "border": "#dc3545", "text": "#721c24", "emoji": "❓"}
                
                html_content += f"""
                <div style="background: {color_scheme['bg']}; border-left: 5px solid {color_scheme['border']}; 
                           padding: 15px; margin: 10px 0; border-radius: 0 10px 10px 0;">
                    <h4 style="color: {color_scheme['text']}; margin: 0 0 15px 0;">
                        {color_scheme['emoji']} {model_name}
                    </h4>
                    <p style="color: {color_scheme['text']}; margin: 0 0 10px 0;">
                        <strong>Classes available:</strong> {result['num_classes']:,}
                    </p>
                    <h5 style="color: {color_scheme['text']}; margin: 0 0 10px 0;">Top 5 Predictions:</h5>
                """
                
                for i, pred in enumerate(result['predictions'][:5], 1):
                    bar_width = int(pred['confidence'] * 100)
                    html_content += f"""
                    <div style="margin: 8px 0; font-family: monospace;">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <span style="color: {color_scheme['text']}; font-weight: bold;">
                                {i}. {pred['class'][:30]}{'...' if len(pred['class']) > 30 else ''}
                            </span>
                            <span style="color: {color_scheme['text']}; font-weight: bold;">
                                {pred['confidence']:.4f}
                            </span>
                        </div>
                        <div style="background: #e9ecef; height: 8px; border-radius: 4px; margin-top: 3px;">
                            <div style="background: {color_scheme['border']}; height: 100%; 
                                       width: {bar_width}%; border-radius: 4px; transition: width 0.3s ease;">
                            </div>
                        </div>
                    </div>
                    """
                
                html_content += "</div>"
        
        html_content += "</div>"
        return HTML(html_content)
    
    @staticmethod
    def export_success_card(output_dir):
        """Create export success notification"""
        return HTML(f"""
        <div style="background: {NotebookStyles.COLORS['success']['bg']}; 
                   border: 1px solid {NotebookStyles.COLORS['success']['border']}; 
                   border-radius: 10px; padding: 15px; margin: 10px 0;">
            <h4 style="color: {NotebookStyles.COLORS['success']['text']}; margin: 0 0 10px 0;">
                ✅ Export Complete!
            </h4>
            <p style="color: {NotebookStyles.COLORS['success']['text']}; margin: 0;">
                Results saved to: <code>{output_dir}</code><br>
                📄 JSON format for further analysis<br>
                📊 Visualizations included
            </p>
        </div>
        """)
    
    @staticmethod
    def final_summary(results):
        """Create final session summary"""
        successful_models = [name for name, result in results.items() if 'error' not in result]
        timestamp = datetime.now().strftime("%H:%M")
        
        return HTML(f"""
        <div style="background: linear-gradient(135deg, #6f42c1, #e83e8c); 
                   color: white; padding: 30px; border-radius: 20px; margin: 30px 0; text-align: center;">
            <h2 style="margin: 0 0 15px 0;">🎉 Analysis Complete!</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                       gap: 20px; margin-top: 20px;">
                <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
                    <h3 style="margin: 0; font-size: 2em;">{len(successful_models)}</h3>
                    <p style="margin: 5px 0 0 0;">Models Classified Successfully</p>
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
                    <h3 style="margin: 0; font-size: 2em;">{timestamp}</h3>
                    <p style="margin: 5px 0 0 0;">Session Completed</p>
                </div>
            </div>
            <p style="margin-top: 20px; opacity: 0.9; font-style: italic;">
                Thank you for using the Satellite Image Classification Demo! 🚀
            </p>
        </div>
        """)
    
    @staticmethod
    def no_results_warning():
        """Display warning when no results are available"""
        return HTML("""
        <div style="background: #fff3cd; border: 1px solid #ffeaa7; 
                   border-radius: 10px; padding: 15px; margin: 20px 0; text-align: center;">
            <h4 style="color: #856404; margin: 0 0 10px 0;">
                ⚠️ No Results to Display
            </h4>
            <p style="color: #856404; margin: 0;">
                Please run the classification step first to see detailed results.
            </p>
        </div>
        """)


# Convenience function to display styles
def display_style(style_func, *args, **kwargs):
    """Helper function to display any style component"""
    display(style_func(*args, **kwargs))


# Quick access to commonly used styles
styles = NotebookStyles()