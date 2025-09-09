"""
Streamlit Web Application for LLM Study & Agent Platform

A user-friendly web interface for interacting with LLMs, agents, and PDF Q&A.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from llm_study.fundamentals import LLMBasics
from agents.core import AgentConfig, AgentManager
from pdf_qa import PDFQASystem
import tempfile


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="LLM Study & Agent Platform",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main header
    st.title("🚀 LLM Study & Agent Development Platform")
    st.markdown("**Learn LLMs, Build Agents, Deploy AI Systems**")
    
    # Sidebar navigation
    st.sidebar.title("📚 Navigation")
    
    page = st.sidebar.selectbox(
        "Choose a module:",
        [
            "🏠 Home",
            "📖 LLM Fundamentals", 
            "🤖 AI Agents",
            "📄 PDF Q&A",
            "🔧 Tools & Utilities",
            "📊 System Status"
        ]
    )
    
    # Route to different pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "📖 LLM Fundamentals":
        show_llm_fundamentals()
    elif page == "🤖 AI Agents":
        show_agents_page()
    elif page == "📄 PDF Q&A":
        show_pdf_qa_page()
    elif page == "🔧 Tools & Utilities":
        show_tools_page()
    elif page == "📊 System Status":
        show_status_page()


def show_home_page():
    """Display the home page."""
    st.header("Welcome to Your LLM Learning Journey! 🎉")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 What You Can Do")
        st.markdown("""
        - **Learn LLM Basics**: Understand tokenization, attention, and embeddings
        - **Build AI Agents**: Create intelligent agents with custom tools
        - **PDF Q&A Systems**: Chat with your documents using RAG
        - **Deploy Solutions**: Production-ready deployment configurations
        """)
        
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. Start with **LLM Fundamentals** to understand the basics
        2. Try **PDF Q&A** to upload and chat with documents
        3. Build **AI Agents** with custom capabilities
        4. Explore **Tools & Utilities** for advanced features
        """)
    
    with col2:
        st.subheader("📈 Learning Path")
        
        # Progress tracking (mock data)
        progress_data = {
            "LLM Fundamentals": 85,
            "Agent Development": 60,
            "PDF Processing": 40,
            "Deployment": 20
        }
        
        for topic, progress in progress_data.items():
            st.metric(topic, f"{progress}%", f"+{progress//10}")
        
        st.subheader("🔗 Quick Links")
        st.markdown("""
        - [GitHub Repository](https://github.com/svenu96/AgenLLMTests)
        - [Documentation](docs/)
        - [Examples](examples/)
        - [API Reference](docs/api_reference.md)
        """)
    
    # Recent activity
    st.subheader("📊 Platform Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Available Models", "5", "+2")
    with col2:
        st.metric("Active Agents", "0", "0")
    with col3:
        st.metric("Documents Processed", "0", "0")
    with col4:
        st.metric("API Calls Today", "0", "0")


def show_llm_fundamentals():
    """Display LLM fundamentals page."""
    st.header("📖 LLM Fundamentals")
    
    tab1, tab2, tab3 = st.tabs(["🔤 Tokenization", "🎯 Attention", "🎨 Embeddings"])
    
    with tab1:
        st.subheader("Understanding Tokenization")
        
        # Text input for tokenization
        text_input = st.text_area(
            "Enter text to tokenize:",
            value="Machine learning is revolutionizing artificial intelligence!",
            height=100
        )
        
        if st.button("Tokenize Text"):
            try:
                llm = LLMBasics()
                if llm.tokenizer:
                    result = llm.tokenization_demo(text_input)
                    
                    st.success("✅ Tokenization completed!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Text:**")
                        st.code(result['original_text'])
                        
                        st.write("**Tokens:**")
                        st.json(result['tokens'])
                    
                    with col2:
                        st.write("**Token IDs:**")
                        st.json(result['token_ids'])
                        
                        st.metric("Vocabulary Size", f"{result['vocab_size']:,}")
                        st.metric("Token Count", len(result['tokens']))
                else:
                    st.error("❌ Tokenizer not available")
                    
            except Exception as e:
                st.error(f"Error: {e}")
    
    with tab2:
        st.subheader("Attention Mechanisms")
        st.info("Attention helps models focus on relevant parts of the input.")
        
        # Attention visualization would go here
        st.markdown("""
        **Key Concepts:**
        - **Queries**: What information am I looking for?
        - **Keys**: What information do I contain?
        - **Values**: What information can I provide?
        - **Attention Weights**: How much focus to give each element
        """)
        
        if st.button("Demo Attention Mechanism"):
            from llm_study.fundamentals import TransformerConcepts
            concepts = TransformerConcepts()
            result = concepts.attention_mechanism_demo()
            
            st.write("**Attention Demo Results:**")
            st.write(f"Words analyzed: {result['words']}")
            st.write("Check the console output for detailed attention calculations!")
    
    with tab3:
        st.subheader("Word Embeddings")
        st.info("Embeddings convert words into numerical vectors that capture meaning.")
        
        # Word similarity demo
        words_input = st.text_input(
            "Enter words separated by commas:",
            value="happy,joy,sad,angry,excited"
        )
        
        if st.button("Analyze Word Embeddings"):
            try:
                words = [w.strip() for w in words_input.split(',') if w.strip()]
                
                llm = LLMBasics()
                if llm.model:
                    result = llm.embedding_exploration(words)
                    st.success("✅ Embedding analysis completed!")
                    st.write("Check the console for similarity visualizations!")
                else:
                    st.error("❌ Model not available")
                    
            except Exception as e:
                st.error(f"Error: {e}")


def show_agents_page():
    """Display AI agents page."""
    st.header("🤖 AI Agents")
    
    # Initialize session state for agents
    if 'agent_manager' not in st.session_state:
        st.session_state.agent_manager = AgentManager()
        st.session_state.current_agent = None
    
    tab1, tab2, tab3 = st.tabs(["➕ Create Agent", "💬 Chat", "📊 Manage"])
    
    with tab1:
        st.subheader("Create New Agent")
        
        with st.form("create_agent_form"):
            agent_name = st.text_input("Agent Name", value="MyAgent")
            agent_description = st.text_area("Description", value="A helpful AI assistant")
            
            agent_type = st.selectbox("Agent Type", ["simple", "tool"])
            
            if agent_type == "tool":
                available_tools = ["calculator", "datetime", "search", "weather"]
                selected_tools = st.multiselect("Select Tools", available_tools)
            else:
                selected_tools = []
            
            system_prompt = st.text_area(
                "System Prompt",
                value="You are a helpful AI assistant.",
                height=100
            )
            
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
            
            submitted = st.form_submit_button("Create Agent")
            
            if submitted:
                try:
                    config = AgentConfig(
                        name=agent_name,
                        description=agent_description,
                        system_prompt=system_prompt,
                        tools=selected_tools,
                        temperature=temperature,
                        verbose=True
                    )
                    
                    agent = st.session_state.agent_manager.create_agent(config, agent_type)
                    st.session_state.current_agent = agent
                    
                    st.success(f"✅ Agent '{agent_name}' created successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error creating agent: {e}")
    
    with tab2:
        st.subheader("Chat with Agent")
        
        # Agent selection
        agents = st.session_state.agent_manager.list_agents()
        
        if agents:
            agent_names = [agent['name'] for agent in agents]
            selected_agent_name = st.selectbox("Select Agent", agent_names)
            
            # Find the selected agent
            selected_agent = None
            for agent_id, agent in st.session_state.agent_manager.agents.items():
                if agent.config.name == selected_agent_name:
                    selected_agent = agent
                    break
            
            if selected_agent:
                # Chat interface
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                # Display chat history
                for message in st.session_state.chat_history:
                    if message['role'] == 'user':
                        st.write(f"🧑 **You:** {message['content']}")
                    else:
                        st.write(f"🤖 **{selected_agent.config.name}:** {message['content']}")
                
                # Chat input
                user_message = st.text_input("Your message:", key="chat_input")
                
                if st.button("Send") and user_message:
                    try:
                        # Add user message to history
                        st.session_state.chat_history.append({
                            'role': 'user',
                            'content': user_message
                        })
                        
                        # Get agent response
                        response = selected_agent.process_message(user_message)
                        
                        # Add agent response to history
                        st.session_state.chat_history.append({
                            'role': 'assistant', 
                            'content': response
                        })
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        st.info("💡 Make sure you have configured your OpenAI API key")
        else:
            st.info("No agents created yet. Go to the 'Create Agent' tab to get started!")
    
    with tab3:
        st.subheader("Manage Agents")
        
        agents = st.session_state.agent_manager.list_agents()
        
        if agents:
            for agent in agents:
                with st.expander(f"🤖 {agent['name']} ({agent['status']})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**ID:** {agent['id'][:8]}...")
                        st.write(f"**Model:** {agent['model']}")
                        st.write(f"**Created:** {agent['created_at'][:19]}")
                    
                    with col2:
                        st.write(f"**Tools:** {', '.join(agent['tools']) if agent['tools'] else 'None'}")
                        st.write(f"**Memory:** {agent['memory_type']}")
                        st.write(f"**Messages:** {agent['conversation_length']}")
                    
                    if st.button(f"Remove {agent['name']}", key=f"remove_{agent['id']}"):
                        st.session_state.agent_manager.remove_agent(agent['id'])
                        st.success(f"Agent '{agent['name']}' removed!")
                        st.rerun()
        else:
            st.info("No agents found.")


def show_pdf_qa_page():
    """Display PDF Q&A page."""
    st.header("📄 PDF Question & Answer")
    
    # Initialize PDF QA system
    if 'pdf_qa_system' not in st.session_state:
        st.session_state.pdf_qa_system = PDFQASystem()
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload PDF", "❓ Ask Questions", "📊 Documents"])
    
    with tab1:
        st.subheader("Upload PDF Document")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            document_id = st.text_input("Document ID", value=uploaded_file.name[:-4])
            
            if st.button("Process PDF"):
                try:
                    with st.spinner("Processing PDF..."):
                        doc_id = st.session_state.pdf_qa_system.add_pdf(tmp_file_path, document_id)
                    
                    st.success(f"✅ PDF processed successfully! Document ID: {doc_id}")
                    
                    # Show document summary
                    summary = st.session_state.pdf_qa_system.get_document_summary(doc_id)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Text Length", f"{summary['text_length']:,} chars")
                        st.metric("Number of Chunks", summary['num_chunks'])
                    with col2:
                        st.metric("Avg Chunk Length", f"{summary['avg_chunk_length']:.0f} chars")
                    
                    # Clean up temp file
                    os.unlink(tmp_file_path)
                    
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {e}")
                    os.unlink(tmp_file_path)
    
    with tab2:
        st.subheader("Ask Questions")
        
        if st.session_state.pdf_qa_system.is_ready:
            question = st.text_area("Enter your question:", height=100)
            
            col1, col2 = st.columns(2)
            with col1:
                num_chunks = st.slider("Number of relevant chunks", 1, 10, 3)
            
            if st.button("Get Answer") and question:
                try:
                    with st.spinner("Generating answer..."):
                        result = st.session_state.pdf_qa_system.ask_question(question, num_chunks)
                    
                    st.subheader("📝 Answer")
                    st.write(result['answer'])
                    
                    st.subheader("📊 Confidence")
                    st.progress(result['confidence'])
                    st.write(f"Confidence Score: {result['confidence']:.2%}")
                    
                    st.subheader("📚 Sources")
                    for i, source in enumerate(result['sources'], 1):
                        with st.expander(f"Source {i} - {source['document_id']} (Score: {source['relevance_score']:.3f})"):
                            st.write(source['text_preview'])
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {e}")
                    st.info("💡 Make sure you have configured your OpenAI API key")
        else:
            st.info("Please upload and process a PDF document first.")
    
    with tab3:
        st.subheader("Document Library")
        
        documents = st.session_state.pdf_qa_system.list_documents()
        
        if documents:
            for doc in documents:
                with st.expander(f"📄 {doc['document_id']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Path:** {doc['path']}")
                        st.write(f"**Text Length:** {doc['text_length']:,} characters")
                    
                    with col2:
                        st.write(f"**Chunks:** {doc['num_chunks']}")
                        st.write(f"**Avg Chunk Size:** {doc['avg_chunk_length']:.0f} chars")
        else:
            st.info("No documents processed yet.")


def show_tools_page():
    """Display tools and utilities page."""
    st.header("🔧 Tools & Utilities")
    
    tab1, tab2, tab3 = st.tabs(["🧮 Calculator", "🕐 DateTime", "🔍 Search"])
    
    with tab1:
        st.subheader("Calculator Tool")
        
        expression = st.text_input("Enter mathematical expression:", value="2 + 3 * 4")
        
        if st.button("Calculate"):
            from agents.tools import CalculatorTool
            calc = CalculatorTool()
            result = calc.execute(expression=expression)
            st.write(result)
    
    with tab2:
        st.subheader("DateTime Tool")
        
        format_type = st.selectbox("Format", ["default", "iso", "custom"])
        
        if format_type == "custom":
            custom_format = st.text_input("Custom format", value="%Y-%m-%d %H:%M:%S")
        else:
            custom_format = None
        
        if st.button("Get Current Time"):
            from agents.tools import DateTimeTool
            dt = DateTimeTool()
            result = dt.execute(format_type=format_type, custom_format=custom_format)
            st.write(result)
    
    with tab3:
        st.subheader("Search Tool (Mock)")
        
        query = st.text_input("Search query:", value="artificial intelligence")
        limit = st.slider("Number of results", 1, 5, 3)
        
        if st.button("Search"):
            from agents.tools import SearchTool
            search = SearchTool()
            result = search.execute(query=query, limit=limit)
            st.write(result)


def show_status_page():
    """Display system status page."""
    st.header("📊 System Status")
    
    # System information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 System Information")
        
        import platform
        import psutil
        
        st.write(f"**Platform:** {platform.system()} {platform.release()}")
        st.write(f"**Python Version:** {platform.python_version()}")
        st.write(f"**CPU Usage:** {psutil.cpu_percent()}%")
        st.write(f"**Memory Usage:** {psutil.virtual_memory().percent}%")
    
    with col2:
        st.subheader("📚 Library Status")
        
        # Check library availability
        libraries = {
            "transformers": "🤖 Transformers",
            "torch": "🔥 PyTorch", 
            "langchain": "🦜 LangChain",
            "streamlit": "⚡ Streamlit",
            "openai": "🤖 OpenAI"
        }
        
        for lib, display_name in libraries.items():
            try:
                __import__(lib)
                st.write(f"✅ {display_name}")
            except ImportError:
                st.write(f"❌ {display_name}")
    
    # Environment variables
    st.subheader("🔐 Environment Configuration")
    
    env_vars = ["OPENAI_API_KEY", "HUGGINGFACE_API_TOKEN"]
    
    for var in env_vars:
        if os.getenv(var):
            st.write(f"✅ {var}: Configured")
        else:
            st.write(f"❌ {var}: Not set")
    
    # Recent activity (mock data)
    st.subheader("📈 Recent Activity")
    
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate mock activity data
    dates = [datetime.now() - timedelta(days=x) for x in range(7, 0, -1)]
    activity_data = pd.DataFrame({
        'Date': dates,
        'API Calls': np.random.randint(10, 100, 7),
        'Documents Processed': np.random.randint(0, 10, 7),
        'Agents Created': np.random.randint(0, 5, 7)
    })
    
    st.line_chart(activity_data.set_index('Date'))


if __name__ == "__main__":
    main()