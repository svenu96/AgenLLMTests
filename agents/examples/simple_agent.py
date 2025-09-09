"""
Simple Agent Example

This example demonstrates how to create and use a basic AI agent
using the agent framework.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.core import AgentConfig, AgentManager
from agents.tools import CalculatorTool, DateTimeTool, SearchTool
import time


def create_simple_agent():
    """Create a simple conversational agent."""
    print("🤖 Creating a Simple Agent")
    print("=" * 40)
    
    # Create agent configuration
    config = AgentConfig(
        name="Assistant",
        description="A helpful AI assistant for general conversations",
        system_prompt="You are a friendly and helpful AI assistant. Be conversational and engaging while providing accurate information.",
        temperature=0.7,
        max_tokens=500,
        verbose=True
    )
    
    # Create agent manager and add agent
    manager = AgentManager()
    agent = manager.create_agent(config, "simple")
    
    return agent, manager


def create_tool_agent():
    """Create an agent with tools."""
    print("\n🔧 Creating a Tool Agent")
    print("=" * 40)
    
    # Create agent configuration with tools
    config = AgentConfig(
        name="ToolBot",
        description="An AI assistant with access to various tools",
        system_prompt="""You are an AI assistant with access to tools. 
        Use tools when appropriate to provide accurate and helpful responses.
        Available tools:
        - calculator: for mathematical calculations
        - datetime: for current date and time
        - search: for finding information (mock)""",
        tools=["calculator", "datetime", "search"],
        temperature=0.3,
        max_tokens=600,
        verbose=True
    )
    
    # Create agent manager and add agent
    manager = AgentManager()
    agent = manager.create_agent(config, "tool")
    
    return agent, manager


def demo_conversation(agent, agent_name):
    """Demonstrate a conversation with an agent."""
    print(f"\n💬 Conversation with {agent_name}")
    print("=" * 40)
    
    # Sample questions to test the agent
    questions = [
        "Hello! How are you today?",
        "What's the weather like?",  # Should trigger search tool if available
        "Can you calculate 15 * 23 + 47?",  # Should trigger calculator tool
        "What time is it?",  # Should trigger datetime tool
        "Tell me a fun fact about artificial intelligence"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n🧑 User: {question}")
        
        try:
            response = agent.process_message(question)
            print(f"🤖 {agent.config.name}: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 This might be due to missing OpenAI API key")
        
        # Add a small delay between questions
        time.sleep(1)
        
        # Stop after first question if API is not available
        if i == 1 and "not available" in response.lower():
            print("\n💡 Stopping demo - API not configured")
            break


def demo_agent_management():
    """Demonstrate agent management features."""
    print("\n📊 Agent Management Demo")
    print("=" * 40)
    
    manager = AgentManager()
    
    # Create multiple agents
    configs = [
        AgentConfig(name="Helper", description="General helper"),
        AgentConfig(name="Calculator", description="Math specialist", tools=["calculator"]),
        AgentConfig(name="TimeKeeper", description="Time specialist", tools=["datetime"])
    ]
    
    agents = []
    for config in configs:
        agent_type = "tool" if config.tools else "simple"
        agent = manager.create_agent(config, agent_type)
        agents.append(agent)
    
    # List all agents
    print("\n📋 Agent Status:")
    for status in manager.list_agents():
        print(f"- {status['name']}: {status['status']} ({len(status['tools'])} tools)")
    
    # Broadcast a message to all agents
    print(f"\n📢 Broadcasting Message:")
    question = "What can you help me with?"
    print(f"Question: {question}")
    
    responses = manager.broadcast_message(question)
    
    for agent_name, response in responses.items():
        print(f"\n🤖 {agent_name}:")
        print(f"   {response[:100]}{'...' if len(response) > 100 else ''}")


def interactive_mode():
    """Run interactive mode for testing agents."""
    print("\n🎮 Interactive Mode")
    print("=" * 40)
    print("Type 'quit' to exit, 'switch' to change agents")
    
    # Create agents
    simple_agent, _ = create_simple_agent()
    tool_agent, _ = create_tool_agent()
    
    current_agent = simple_agent
    current_name = "Simple Agent"
    
    print(f"\n💬 Chatting with {current_name}")
    print("Type your message:")
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'switch':
                # Switch between agents
                if current_agent == simple_agent:
                    current_agent = tool_agent
                    current_name = "Tool Agent"
                else:
                    current_agent = simple_agent
                    current_name = "Simple Agent"
                print(f"🔄 Switched to {current_name}")
                continue
            elif not user_input:
                print("💡 Please enter a message or 'quit' to exit")
                continue
            
            # Get response from current agent
            response = current_agent.process_message(user_input)
            print(f"🤖 {current_agent.config.name}: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 This might be due to missing API configuration")


def main():
    """Main function to run the agent examples."""
    print("🚀 Simple Agent Examples")
    print("=" * 50)
    
    print("\n📚 This example demonstrates:")
    print("1. Creating simple conversational agents")
    print("2. Creating agents with tools")
    print("3. Agent conversation capabilities")
    print("4. Agent management features")
    
    try:
        # Create and demo simple agent
        simple_agent, _ = create_simple_agent()
        demo_conversation(simple_agent, "Simple Agent")
        
        # Create and demo tool agent
        tool_agent, _ = create_tool_agent()
        demo_conversation(tool_agent, "Tool Agent")
        
        # Demo agent management
        demo_agent_management()
        
        # Ask if user wants interactive mode
        print(f"\n🎮 Would you like to try interactive mode? (y/n)")
        choice = input().strip().lower()
        
        if choice in ['y', 'yes']:
            interactive_mode()
        else:
            print("✅ Demo completed!")
            
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print("\n💡 Common issues:")
        print("- Missing OpenAI API key (set OPENAI_API_KEY environment variable)")
        print("- Missing dependencies (run: pip install -r requirements.txt)")
        print("- Network connectivity issues")
    
    print(f"\n🎉 Thanks for trying the agent examples!")
    print("🚀 Next steps:")
    print("- Try the PDF Q&A example")
    print("- Explore multi-agent systems")
    print("- Build custom tools for your agents")


if __name__ == "__main__":
    main()