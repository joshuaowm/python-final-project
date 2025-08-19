"""
Satellite Image Analysis Hub - Main Application Entry Point
Multipage Streamlit App with Navigation
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

# Page config - this should be the first Streamlit command
st.set_page_config(
    page_title="Satellite Image Analysis Hub",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
        font-style: italic;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .feature-card h3 {
        color: white;
        margin-bottom: 1rem;
        font-size: 1.8rem;
    }
    .feature-card p {
        color: #f0f0f0;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .navigation-card {
        background: white;
        padding: 1rem 2rem;
        border-radius: 15px;
        border: 2px solid #1f77b4;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .navigation-card h3 {
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .tech-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
    }
    .use-case-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .model-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .model-item {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #d0e7f0;
        margin-bottom: 0.5rem;
    }
    .stats-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }
    .workflow-step {
        background: white;
        padding: 0.6rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #e0e0e0;
        position: relative;
    }
    .workflow-number {
        position: absolute;
        top: -15px;
        left: 20px;
        background: #1f77b4;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
    }
    .sidebar-nav {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    strong {
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar navigation
    st.sidebar.markdown("""
    <div class="sidebar-nav">
        <h2 style="color: #1f77b4; text-align: center; margin-bottom: 1rem;">📍 Navigation</h2>
        <p style="text-align: center; color: #666; margin-bottom: 1rem;">
            Welcome to the Satellite Image Analysis Hub! Use the pages above to analyze your satellite and aerial imagery.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 📄 Available Pages:")
    st.sidebar.markdown("""
    - **🏠 Home** - Overview and information (current page)
    - **🎯 Classification** - Classify land cover types
    - **🗺️ Segmentation** - Pixel-level image segmentation
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 🚀 Quick Start:
    1. Choose **Classification** for land cover identification
    2. Choose **Segmentation** for detailed pixel mapping
    3. Upload your satellite image
    4. Select models and run analysis
    """)
    
    # Main content
    # Header
    st.markdown('<h1 class="main-header">🛰️ Satellite Image Analysis Hub</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered platform for rapid satellite and aerial image analysis. Instantly classify and segment images with multiple state-of-the-art models to gain quick insights</p>', unsafe_allow_html=True)
    
    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
                
    # Demo visualization
    st.markdown("## 🔦 What This Application Does")
    st.image("results/example.jpg", caption="Segmentation and Classification Results Example", use_container_width=True)
    
    st.markdown('<div style="height:25px;"></div>', unsafe_allow_html=True) # Gap
    
    # Navigation cards
    st.markdown("## 👇 Choose Your Analysis Type")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="navigation-card">
            <h3>🎯 Image Classification</h3>
            <p style="color: #666; margin-bottom: 1.5rem;">Identify what land cover types are present in your satellite images using multiple deep learning models.</p>
            <p><strong>Perfect for:</strong></p>
            <ul style="text-align: left; color: #666;">
                <li>Land cover mapping</li>
                <li>Agricultural monitoring</li>
                <li>Environmental assessment</li>
                <li>Urban planning</li>
            </ul>
            <p style="color: #666; margin-top: 1rem;"><strong>🔍 Go to Classification page to get started!</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="navigation-card">
            <h3>🗺️ Semantic Segmentation</h3>
            <p style="color: #666; margin-bottom: 1.5rem;">Perform pixel-level analysis to create detailed maps of different regions and objects in your images.</p>
            <p><strong>Perfect for:</strong></p>
            <ul style="text-align: left; color: #666;">
                <li>Detailed land use mapping</li>
                <li>Infrastructure analysis</li>
                <li>Precision agriculture</li>
                <li>Change detection</li>
            </ul>
            <p style="color: #666; margin-top: 1rem;"><strong>🔍 Go to Segmentation page to get started!</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div style="height:25px;"></div>', unsafe_allow_html=True) # Gap
    
    # Application workflow
    st.markdown("## 📄 How It Works")
    
    workflow_steps = [
        {
            "title": "Navigate to Analysis Page",
            "description": "Choose Classification or Segmentation from the sidebar based on your analysis needs."
        },
        {
            "title": "Upload Your Image",
            "description": "Upload satellite or aerial images in various formats (PNG, JPG, JPEG, TIFF) or provide image URLs."
        },
        {
            "title": "Select Models",
            "description": "Choose from multiple pre-trained models specialized for different types of satellite imagery analysis."
        },
        {
            "title": "Run Analysis & View Results",
            "description": "Get detailed results with confidence scores, visualizations, and comprehensive analysis of your imagery."
        }
    ]
    
    for i, step in enumerate(workflow_steps, 1):
        st.markdown(f"""
        <div class="workflow-step">
            <div class="workflow-number">{i}</div>
            <h4 style="margin-top: 10px; color: #1f77b4;">{step['title']}</h4>
            <p style="color: #666; line-height: 1.6;">{step['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div style="height:25px;"></div>', unsafe_allow_html=True) # Gap
    
    # Available models section
    st.markdown("## 🧠 Available Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="text-align: center;">
            <h4>🎯 Classification Models</h4>
        </div>
        """, unsafe_allow_html=True)
        classification_models = [
            "EuroSAT Models - 10-class European land cover imagery",
            "Land Cover Models - 8-class global land cover detection", 
            "Aerial Models - 12-class aerial scene classification",
            "ImageNet Models - ImageNet-based, adapted for remote sensing"
        ]
        
        for model in classification_models:
            st.markdown(f"""
            <div class="model-item">
                <strong style="color: #000;">{model.split(' - ')[0]}</strong><br>
                <small style="color: #666;">{model.split(' - ')[1]}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h4>🗺️ Segmentation Models</h4>
        </div>
        """, unsafe_allow_html=True)
        segmentation_models = [
            "SegFormer - Transformer-based segmentation for high accuracy",
            "BEiT - Pre-trained vision transformer for dense prediction",
            "DPT - Transformer for fine-grained depth & segmentation",
            "UperNet - Multi-scale parsing for detailed scene understanding",
            "SMP - PyTorch library with diverse segmentation architectures"
        ]
        
        for model in segmentation_models:
            st.markdown(f"""
            <div class="model-item">
                <strong style="color: #000;">{model.split(' - ')[0]}</strong><br>
                <small style="color: #666;">{model.split(' - ')[1]}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div style="height:25px;"></div>', unsafe_allow_html=True) # Gap
    
    # Use cases
    st.markdown("## 🌍 Use Cases & Applications")
    
    use_cases = [
        {
            "title": "🌱 Environmental Monitoring",
            "description": "Track deforestation, monitor ecosystem health, assess environmental changes over time."
        },
        {
            "title": "🏙️ Urban Planning", 
            "description": "Analyze urban development, map land use patterns, monitor infrastructure growth."
        },
        {
            "title": "🌾 Agriculture",
            "description": "Monitor crop health, optimize irrigation, assess field conditions and yields."
        },
        {
            "title": "🚨 Disaster Response",
            "description": "Assess disaster damage, monitor flood areas, track wildfire spread."
        },
        {
            "title": "📊 Research & Analysis",
            "description": "Academic research, climate studies, geographic analysis, education."
        },
        {
            "title": "🏢 Commercial Applications",
            "description": "Real estate analysis, resource exploration, business intelligence."
        }
    ]
    
    for i in range(0, len(use_cases), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(use_cases):
                case = use_cases[i]
                st.markdown(f"""
                <div class="use-case-card">
                    <h4 style="color: #1f77b4; margin-bottom: 0.5rem;">{case['title']}</h4>
                    <p style="color: #666; margin: 0; line-height: 1.5;">{case['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if i + 1 < len(use_cases):
                case = use_cases[i + 1]
                st.markdown(f"""
                <div class="use-case-card">
                    <h4 style="color: #1f77b4; margin-bottom: 0.5rem;">{case['title']}</h4>
                    <p style="color: #666; margin: 0; line-height: 1.5;">{case['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div style="height:25px;"></div>', unsafe_allow_html=True) # Gap
    
    # Technical specifications
    st.markdown("""
    <div class="tech-section">
        <h2 style="color: #1f77b4; margin-bottom: 1.5rem;">⚙️ Technical Specifications</h2>
        <div class="model-grid">
            <div class="model-item">
                <h4 style="color: #1f77b4; margin-bottom: 0.5rem;">🔥 Framework</h4>
                <p style="color: #666; margin: 0;">PyTorch with CUDA acceleration</p>
            </div>
            <div class="model-item">
                <h4 style="color: #1f77b4; margin-bottom: 0.5rem;">🖼️ Formats</h4>
                <p style="color: #666; margin: 0;">PNG, JPEG, TIFF, URL inputs</p>
            </div>
            <div class="model-item">
                <h4 style="color: #1f77b4; margin-bottom: 0.5rem;">🚀 Processing</h4>
                <p style="color: #666; margin: 0;">GPU-accelerated inference</p>
            </div>
            <div class="model-item">
                <h4 style="color: #1f77b4; margin-bottom: 0.5rem;">📊 Models</h4>
                <p style="color: #666; margin: 0;">15+ specialized models</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="height:25px;"></div>', unsafe_allow_html=True) # Gap
    
    # Statistics
    st.markdown("""
    <div class="stats-container">
        <h2 style="margin-bottom: 1.5rem;">📈 Platform Statistics</h2>
        <div class="model-grid" style="color: white;">
            <div>
                <h3 style="color: white; font-size: 2rem; margin: 0;">15+</h3>
                <p style="margin: 0; opacity: 0.9;">Pre-trained Models</p>
            </div>
            <div>
                <h3 style="color: white; font-size: 2rem; margin: 0;">2</h3>
                <p style="margin: 0; opacity: 0.9;">Analysis Types</p>
            </div>
            <div>
                <h3 style="color: white; font-size: 2rem; margin: 0;">160+</h3>
                <p style="margin: 0; opacity: 0.9;">Land Cover Classes</p>
            </div>
            <div>
                <h3 style="color: white; font-size: 2rem; margin: 0;">∞</h3>
                <p style="margin: 0; opacity: 0.9;">Image Analyses</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
        
    # Footer
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🛰️ Satellite Image Analysis Hub**  
        Advanced AI for Earth Observation
        """)
    
    with col2:
        st.markdown("""
        **🔧 Built With**  
        PyTorch • Streamlit • Transformers  
        Matplotlib • PIL • NumPy
        """)
    
    with col3:
        st.markdown("""
        **🚀 Features**  
        Multi-Model Analysis  
        Real-time Processing  
        Interactive Visualizations
        """)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <p style="margin: 0; color: #666;">
            Ready to analyze satellite imagery? Choose <strong>Classification</strong> or <strong>Segmentation</strong> from the sidebar to get started! 🚀
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()