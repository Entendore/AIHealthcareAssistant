import streamlit as st
import os
import json
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from typing import Dict, List, Any, Optional, Tuple
import re
import logging
from dataclasses import dataclass

# ==================== CONFIGURATION ====================
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "meditron-7b"
DIAGNOSIS_COOLDOWN_SECONDS = 60
TIMEOUT_SECONDS = 300

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== AUTOGEN SETUP ====================
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError as e:
    st.error(f"AutoGen not available: {e}")
    AUTOGEN_AVAILABLE = False

# ==================== SESSION STATE ====================
def initialize_session_state():
    """Initialize all session state variables with default values."""
    defaults = {
        'patient_data': {
            'name': '', 'age': '', 'gender': '', 'symptoms': [],
            'medical_history': '', 'medications': '', 'allergies': '', 'additional_notes': ''
        },
        'diagnosis_results': None,
        'agent_status': {
            'symptom_analyzer': 'idle', 'medical_researcher': 'idle',
            'diagnostic_specialist': 'idle', 'treatment_advisor': 'idle',
            'report_generator': 'idle', 'coordinator': 'idle'
        },
        'selected_model': DEFAULT_MODEL,
        'chat_history': [],
        'processing': False,
        'last_diagnosis_time': 0,
        'symptom_input_field': '',
        'current_step': 0,
        'total_steps': 6,
        'patient_records': []
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# ==================== UTILITY FUNCTIONS ====================
def sanitize_input(text: str) -> str:
    """Sanitize input to prevent prompt injection."""
    if not text:
        return ""
    return re.sub(r'[^\w\s.,\-/()\[\]]', '', str(text))[:1000]

def validate_patient_data(data: Dict[str, Any]) -> List[str]:
    """Validate patient data and return list of errors."""
    errors = []
    
    if not data['symptoms']:
        errors.append("❌ At least one symptom is required")
    
    if data['age']:
        try:
            age = int(data['age'])
            if not 0 <= age <= 150:
                errors.append("❌ Age must be between 0 and 150")
        except ValueError:
            errors.append("❌ Age must be a valid number")
    
    return errors

def check_ollama_status() -> Tuple[bool, str]:
    """Check Ollama connection status."""
    try:
        response = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=3)
        if response.status_code == 200:
            return True, "✅ Connected"
        return False, f"❌ HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "❌ Connection refused - Is Ollama running?"
    except requests.exceptions.Timeout:
        return False, "❌ Connection timeout"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def get_ollama_models() -> List[str]:
    """Fetch available models from Ollama."""
    try:
        response = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if models:
                model_names = [model['name'] for model in models]
                meditron_models = [m for m in model_names if 'meditron' in m]
                other_models = [m for m in model_names if 'meditron' not in m]
                return meditron_models + other_models
    except Exception:
        pass
    
    return ['meditron-7b', 'meditron-70b', 'llama2', 'mistral', 'codellama']

def reset_agent_status():
    """Reset all agent statuses to 'idle'."""
    for agent_name in st.session_state.agent_status:
        st.session_state.agent_status[agent_name] = 'idle'
    st.session_state.current_step = 0

def update_agent_status(agent_name: str, status: str):
    """Update status for a specific agent."""
    if agent_name in st.session_state.agent_status:
        st.session_state.agent_status[agent_name] = status
        st.rerun()

def save_patient_record(patient_data: Dict[str, Any], diagnosis: str):
    """Save patient record to history."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "patient_data": patient_data,
        "diagnosis": diagnosis
    }
    st.session_state.patient_records.append(record)
    if len(st.session_state.patient_records) > 10:
        st.session_state.patient_records = st.session_state.patient_records[-10:]

# ==================== AUTOGEN AGENT CREATION ====================
def create_autogen_agents() -> Optional[Dict[str, Any]]:
    """Initialize AutoGen agents with Ollama LLM."""
    if not AUTOGEN_AVAILABLE:
        st.error("AutoGen dependencies not available.")
        return None
    
    try:
        model = st.session_state.selected_model or DEFAULT_MODEL
        
        # Configure LLM for AutoGen
        llm_config = {
            "config_list": [
                {
                    "model": model,
                    "base_url": OLLAMA_BASE_URL,
                    "api_key": "NULL",
                    "temperature": 0.1
                }
            ],
            "timeout": TIMEOUT_SECONDS
        }
        
        # Create agents
        agents = {}
        
        # Symptom Analyzer Agent
        agents['symptom_analyzer'] = AssistantAgent(
            name="SymptomAnalyzer",
            system_message="""You are a medical professional specializing in symptom analysis.
            Analyze patient symptoms, identify patterns, and provide initial insights.
            Focus on:
            1. Symptom patterns and relationships
            2. Affected body systems
            3. Initial observations
            4. Questions for clarification
            
            Be thorough and methodical. Respond professionally.""",
            llm_config=llm_config,
        )
        
        # Medical Researcher Agent
        agents['medical_researcher'] = AssistantAgent(
            name="MedicalResearcher",
            system_message="""You are a medical researcher with extensive knowledge of medical literature.
            Research the symptoms and provide evidence-based insights.
            Focus on:
            1. Relevant medical literature
            2. Recent findings
            3. Common conditions
            4. Evidence-based insights
            
            Cite sources when possible. Respond professionally.""",
            llm_config=llm_config,
        )
        
        # Diagnostic Specialist Agent
        agents['diagnostic_specialist'] = AssistantAgent(
            name="DiagnosticSpecialist",
            system_message="""You are a diagnostic specialist with years of experience.
            Provide differential diagnosis based on symptoms and research.
            Focus on:
            1. Ranked list of possible diagnoses
            2. Likelihood percentages
            3. Supporting factors
            4. Recommended tests
            
            Consider multiple possibilities. Respond professionally.""",
            llm_config=llm_config,
        )
        
        # Treatment Advisor Agent
        agents['treatment_advisor'] = AssistantAgent(
            name="TreatmentAdvisor",
            system_message="""You are a treatment advisor with expertise in treatment plans.
            Recommend evidence-based treatments.
            Focus on:
            1. Treatment options for likely diagnoses
            2. Lifestyle recommendations
            3. Follow-up care
            4. Side effects and contraindications
            
            Consider patient preferences. Respond professionally.""",
            llm_config=llm_config,
        )
        
        # Report Generator Agent
        agents['report_generator'] = AssistantAgent(
            name="ReportGenerator",
            system_message="""You are a medical report generator.
            Create a comprehensive medical report.
            Include:
            1. Patient information summary
            2. Symptom analysis
            3. Research findings
            4. Differential diagnosis
            5. Treatment recommendations
            6. Medical disclaimer
            
            Organize information clearly. Respond professionally.""",
            llm_config=llm_config,
        )
        
        # Coordinator Agent
        agents['coordinator'] = AssistantAgent(
            name="Coordinator",
            system_message="""You are the coordinator of the medical diagnosis team.
            Organize the diagnostic process and ensure all agents contribute.
            Your role:
            1. Direct the process in correct order
            2. Request specific information
            3. Ensure all aspects are covered
            4. Identify gaps in analysis
            
            Maintain professional coordination. Respond professionally.""",
            llm_config=llm_config,
        )
        
        # User Proxy Agent
        agents['user_proxy'] = UserProxyAgent(
            name="UserProxy",
            system_message="You are a proxy for the user, collecting patient information and coordinating the diagnosis process.",
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        
        return agents
        
    except Exception as e:
        st.error(f"Error initializing AutoGen agents: {e}")
        logger.error(f"Error initializing AutoGen agents: {e}")
        return None

def create_patient_summary(patient_data: Dict[str, Any]) -> str:
    """Create a formatted summary of patient data for the agents."""
    sanitized_data = {k: sanitize_input(v) if isinstance(v, str) else v 
                     for k, v in patient_data.items()}
    
    return f"""
PATIENT INFORMATION:
Name: {sanitized_data['name']}
Age: {sanitized_data['age']}
Gender: {sanitized_data['gender']}

SYMPTOMS:
{', '.join(sanitized_data['symptoms'])}

MEDICAL HISTORY:
{sanitized_data['medical_history']}

CURRENT MEDICATIONS:
{sanitized_data['medications']}

ALLERGIES:
{sanitized_data['allergies']}

ADDITIONAL NOTES:
{sanitized_data['additional_notes']}
"""

def run_autogen_chain_sync(patient_data: Dict[str, Any]) -> str:
    """Run the AutoGen agent chain with progress tracking."""
    try:
        agents = create_autogen_agents()
        if not agents:
            return "Error: Could not initialize agents."
        
        # Create group chat
        agent_list = [
            agents['coordinator'],
            agents['symptom_analyzer'],
            agents['medical_researcher'],
            agents['diagnostic_specialist'],
            agents['treatment_advisor'],
            agents['report_generator']
        ]
        
        group_chat = GroupChat(
            agents=agent_list,
            messages=[],
            max_round=20
        )
        
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=agents['coordinator'].llm_config
        )
        
        # Create patient summary
        patient_summary = create_patient_summary(patient_data)
        
        # Start the chat with a clear task
        initial_message = f"""
Please coordinate a comprehensive medical diagnosis for the following patient:

{patient_summary}

Follow this process:
1. SymptomAnalyzer: Analyze symptoms
2. MedicalResearcher: Provide research insights
3. DiagnosticSpecialist: Give differential diagnosis
4. TreatmentAdvisor: Recommend treatments
5. ReportGenerator: Create comprehensive report

Ensure each specialist provides thorough analysis before proceeding.
"""
        
        # Execute the chat
        result = agents['user_proxy'].initiate_chat(
            manager,
            message=initial_message,
            max_turns=30
        )
        
        # Extract the final report
        final_report = ""
        for message in reversed(group_chat.messages):
            if message['name'] == 'ReportGenerator' and message['content'].strip():
                final_report = message['content']
                break
        
        if not final_report:
            final_report = group_chat.messages[-1]['content'] if group_chat.messages else "No report generated."
        
        return final_report
        
    except Exception as e:
        logger.error(f"Error during diagnosis: {str(e)}")
        return f"Error during diagnosis: {str(e)}"

# ==================== MINIMALIST CSS ====================
def load_minimalist_css():
    """Load minimalist CSS for the application."""
    st.markdown("""
    <style>
    /* Global Styles */
    .main {
        padding-top: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header */
    .app-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .app-header h1 {
        color: #1e293b;
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }
    
    .app-header p {
        color: #64748b;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
    }
    
    /* Status Indicators */
    .status-indicator {
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .status-success {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    .status-error {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    .status-warning {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fed7aa;
    }
    
    /* Agent Status */
    .agent-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        background: #f8fafc;
        border-radius: 6px;
        border-left: 3px solid #e2e8f0;
    }
    
    .agent-row.active {
        border-left-color: #3b82f6;
        background: #eff6ff;
    }
    
    .agent-row.completed {
        border-left-color: #10b981;
        background: #f0fdf4;
    }
    
    .agent-name {
        font-weight: 500;
        color: #374151;
    }
    
    .agent-status {
        font-size: 0.875rem;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        background: #e5e7eb;
        color: #6b7280;
    }
    
    .agent-status.running {
        background: #fef3c7;
        color: #92400e;
    }
    
    .agent-status.completed {
        background: #d1fae5;
        color: #065f46;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
    }
    
    .stButton > button:disabled {
        background-color: #9ca3af;
        cursor: not-allowed;
    }
    
    /* Form Elements */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        border-radius: 6px;
        border: 1px solid #d1d5db;
    }
    
    /* Symptoms */
    .symptom-tag {
        display: inline-block;
        background: #e0e7ff;
        color: #3730a3;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        border-radius: 12px;
        font-size: 0.875rem;
    }
    
    /* Results */
    .result-container {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        white-space: pre-wrap;
        font-family: 'Segoe UI', system-ui, sans-serif;
        line-height: 1.6;
        max-height: 500px;
        overflow-y: auto;
    }
    
    /* Disclaimer */
    .disclaimer {
        background: #fffbeb;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #92400e;
        font-size: 0.875rem;
    }
    
    /* Progress */
    .progress-container {
        background: #e5e7eb;
        border-radius: 8px;
        height: 8px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        background: #3b82f6;
        height: 100%;
        transition: width 0.3s ease;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: white;
    }
    
    /* Patient Summary */
    .patient-info {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .info-item {
        display: flex;
        flex-direction: column;
    }
    
    .info-label {
        font-size: 0.875rem;
        color: #6b7280;
        margin-bottom: 0.25rem;
    }
    
    .info-value {
        font-weight: 500;
        color: #1f2937;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== UI COMPONENTS ====================
def render_patient_form():
    """Render the patient information form."""
    st.markdown("### Patient Information")
    
    # Symptoms section
    st.markdown("#### Symptoms")
    
    def symptom_input_callback():
        if st.session_state.symptom_input_field:
            st.session_state.patient_data['symptoms'].append(
                sanitize_input(st.session_state.symptom_input_field)
            )
            st.session_state.symptom_input_field = ''
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input(
            "Enter a symptom",
            key="symptom_input_field",
            on_change=symptom_input_callback
        )
    with col2:
        if st.button("Add", key="add_symptom_btn"):
            st.rerun()
    
    # Display current symptoms
    if st.session_state.patient_data['symptoms']:
        st.markdown("**Current Symptoms:**")
        cols = st.columns(min(len(st.session_state.patient_data['symptoms']), 5))
        for i, symptom in enumerate(st.session_state.patient_data['symptoms']):
            with cols[i % 5]:
                st.markdown(f"<span class='symptom-tag'>{symptom}</span>", unsafe_allow_html=True)
        
        if st.button("Clear All Symptoms", key="clear_symptoms_btn"):
            st.session_state.patient_data['symptoms'] = []
            st.rerun()
    
    st.markdown("---")
    
    # Patient details
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Name", value=st.session_state.patient_data['name'])
            patient_age = st.text_input("Age", value=st.session_state.patient_data['age'])
        with col2:
            patient_gender = st.selectbox(
                "Gender",
                ["", "Male", "Female", "Other"],
                index=["", "Male", "Female", "Other"].index(st.session_state.patient_data['gender'])
                if st.session_state.patient_data['gender'] in ["", "Male", "Female", "Other"] else 0
            )
        
        medical_history = st.text_area("Medical History", value=st.session_state.patient_data['medical_history'])
        medications = st.text_area("Current Medications", value=st.session_state.patient_data['medications'])
        allergies = st.text_area("Allergies", value=st.session_state.patient_data['allergies'])
        
        submitted = st.form_submit_button("Save Information")
        
        if submitted:
            st.session_state.patient_data = {
                'name': sanitize_input(patient_name),
                'age': sanitize_input(patient_age),
                'gender': sanitize_input(patient_gender),
                'symptoms': st.session_state.patient_data['symptoms'],
                'medical_history': sanitize_input(medical_history),
                'medications': sanitize_input(medications),
                'allergies': sanitize_input(allergies),
                'additional_notes': ''
            }
            
            errors = validate_patient_data(st.session_state.patient_data)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                st.success("✅ Information saved")

def render_agent_status():
    """Render agent status indicators."""
    st.markdown("### Agent Status")
    
    agents_config = [
        ('symptom_analyzer', 'Symptom Analyzer'),
        ('medical_researcher', 'Medical Researcher'),
        ('diagnostic_specialist', 'Diagnostic Specialist'),
        ('treatment_advisor', 'Treatment Advisor'),
        ('report_generator', 'Report Generator'),
        ('coordinator', 'Coordinator')
    ]
    
    for agent_key, agent_name in agents_config:
        status = st.session_state.agent_status.get(agent_key, 'idle')
        status_class = status
        
        st.markdown(f"""
        <div class="agent-row {status_class}">
            <span class="agent-name">{agent_name}</span>
            <span class="agent-status {status_class}">{status.title()}</span>
        </div>
        """, unsafe_allow_html=True)

def render_patient_summary():
    """Render patient summary card."""
    st.markdown("### Patient Summary")
    
    with st.container():
        symptoms_display = ', '.join(st.session_state.patient_data['symptoms']) \
                         if st.session_state.patient_data['symptoms'] else 'None provided'
        
        st.markdown(f"""
        <div class="patient-info">
            <div class="info-item">
                <span class="info-label">Name</span>
                <span class="info-value">{st.session_state.patient_data['name'] or 'Not provided'}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Age</span>
                <span class="info-value">{st.session_state.patient_data['age'] or 'Not provided'}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Gender</span>
                <span class="info-value">{st.session_state.patient_data['gender'] or 'Not provided'}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Symptoms</span>
                <span class="info-value">{symptoms_display}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Medical History</span>
                <span class="info-value">{st.session_state.patient_data['medical_history'] or 'None'}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Medications</span>
                <span class="info-value">{st.session_state.patient_data['medications'] or 'None'}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Allergies</span>
                <span class="info-value">{st.session_state.patient_data['allergies'] or 'None'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==================== MAIN DASHBOARD ====================
def display_dashboard():
    """Main dashboard with minimalist design."""
    load_minimalist_css()
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="app-header">
        <h1>MediChain AI</h1>
        <p>Healthcare Diagnosis Assistant with AutoGen Agents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> This AI assistant is for informational purposes only and is not a substitute for professional medical advice.
    </div>
    """, unsafe_allow_html=True)
    
    # Check Ollama status
    ollama_status, status_msg = check_ollama_status()
    
    if not ollama_status:
        st.markdown(f"""
        <div class="status-indicator status-error">
            {status_msg}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <h4>Setup Instructions:</h4>
            <ol>
                <li>Install Ollama: <code>curl -fsSL https://ollama.ai/install.sh | sh</code></li>
                <li>Start Ollama: <code>ollama serve</code></li>
                <li>Pull model: <code>ollama pull meditron-7b</code></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### Configuration")
        
        # Model selection
        available_models = get_ollama_models()
        current_index = available_models.index(st.session_state.selected_model) \
                       if st.session_state.selected_model in available_models else 0
        
        selected_model = st.selectbox(
            "Model",
            available_models,
            index=current_index
        )
        st.session_state.selected_model = selected_model
        
        # Status
        status_class = "status-success" if ollama_status else "status-error"
        st.markdown(f"""
        <div class="status-indicator {status_class}">
            {status_msg}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        render_patient_form()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Patient summary
        render_patient_summary()
        
        # Diagnosis button
        time_since_last = time.time() - st.session_state.last_diagnosis_time
        is_cooled_down = time_since_last >= DIAGNOSIS_COOLDOWN_SECONDS
        
        st.markdown("### Diagnosis")
        
        col_btn1, col_btn2 = st.columns([1, 2])
        with col_btn1:
            btn_disabled = not ollama_status or st.session_state.processing or not is_cooled_down
            diagnosis_btn = st.button(
                "Run Diagnosis",
                disabled=btn_disabled,
                type="primary"
            )
        
        with col_btn2:
            if st.session_state.processing:
                st.info("⏳ Processing...")
            elif not is_cooled_down:
                st.warning(f"⏱️ Wait {DIAGNOSIS_COOLDOWN_SECONDS - int(time_since_last)}s")
        
        # Handle diagnosis
        if diagnosis_btn:
            errors = validate_patient_data(st.session_state.patient_data)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                if not is_cooled_down:
                    st.warning(f"Please wait {DIAGNOSIS_COOLDOWN_SECONDS - int(time_since_last)} seconds.")
                else:
                    reset_agent_status()
                    st.session_state.processing = True
                    
                    with st.spinner("Running AutoGen agents..."):
                        try:
                            result = run_autogen_chain_sync(st.session_state.patient_data)
                            
                            st.session_state.diagnosis_results = result
                            st.session_state.last_diagnosis_time = time.time()
                            st.session_state.chat_history.append({
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "patient": st.session_state.patient_data['name'],
                                "result": result
                            })
                            
                            save_patient_record(st.session_state.patient_data, result)
                            
                            for agent_name in st.session_state.agent_status:
                                update_agent_status(agent_name, 'completed')
                            
                            st.success("✅ Diagnosis completed")
                            
                        except Exception as e:
                            st.error(f"❌ Diagnosis failed: {str(e)}")
                            reset_agent_status()
                        finally:
                            st.session_state.processing = False
        
        # Display results
        if st.session_state.diagnosis_results and not st.session_state.processing:
            st.markdown("### Results")
            st.markdown(f"""
            <div class="result-container">
                <strong>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</strong><br><br>
                {st.session_state.diagnosis_results}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Agent status panel
        render_agent_status()
        
        # Model info
        st.markdown("### Model Info")
        st.markdown(f"""
        <div class="card">
            <p><strong>Model:</strong> {st.session_state.selected_model}</p>
            <p><strong>Framework:</strong> AutoGen</p>
            <p><strong>Provider:</strong> Ollama</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== MAIN APP ====================
def main():
    """Main application entry point."""
    initialize_session_state()
    
    # Simple navigation
    page = st.sidebar.selectbox("Navigation", ["Dashboard", "About"])
    
    if page == "Dashboard":
        display_dashboard()
    elif page == "About":
        load_minimalist_css()
        st.markdown("""
        <div class="app-header">
            <h1>About MediChain AI</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>Our Mission</h3>
            <p>MediChain AI assists healthcare professionals by providing AI-powered diagnostic insights through AutoGen agents.</p>
        </div>
        
        <div class="card">
            <h3>How It Works</h3>
            <ol>
                <li><strong>Coordinator:</strong> Organizes the diagnostic process</li>
                <li><strong>Symptom Analyzer:</strong> Identifies patterns</li>
                <li><strong>Medical Researcher:</strong> Provides evidence</li>
                <li><strong>Diagnostic Specialist:</strong> Differential diagnosis</li>
                <li><strong>Treatment Advisor:</strong> Recommends treatments</li>
                <li><strong>Report Generator:</strong> Creates report</li>
            </ol>
        </div>
        
        <div class="disclaimer">
            <strong>⚠️ Medical Disclaimer:</strong> This AI assistant is for informational purposes only and is not a substitute for professional medical advice.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()