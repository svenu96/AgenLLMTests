"""
Core Agent Framework

Base classes and utilities for building AI agents.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import uuid

from langchain.llms.base import LLM
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage


@dataclass
class AgentConfig:
    """Configuration for an AI agent."""
    name: str
    description: str = ""
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    system_prompt: str = "You are a helpful AI assistant."
    tools: List[str] = field(default_factory=list)
    memory_type: str = "conversation"
    memory_size: int = 10
    verbose: bool = False


class BaseAgent(ABC):
    """
    Base class for all AI agents.
    
    Provides common functionality like memory management,
    tool usage, and conversation handling.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.conversation_history = []
        self.tools = {}
        self.memory = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO if config.verbose else logging.WARNING)
        self.logger = logging.getLogger(f"Agent-{config.name}")
        
        # Initialize components
        self._initialize_llm()
        self._initialize_memory()
        self._initialize_tools()
        
        self.logger.info(f"🤖 Agent '{config.name}' initialized with ID: {self.id}")
    
    def _initialize_llm(self):
        """Initialize the language model."""
        try:
            if "gpt" in self.config.model_name.lower():
                self.llm = ChatOpenAI(
                    model_name=self.config.model_name,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
            else:
                # For other models, you might need different initialization
                self.logger.warning(f"Model {self.config.model_name} not directly supported. Using OpenAI as fallback.")
                self.llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None
    
    def _initialize_memory(self):
        """Initialize memory system based on config."""
        from .memory import ConversationMemory, VectorMemory
        
        if self.config.memory_type == "conversation":
            self.memory = ConversationMemory(max_size=self.config.memory_size)
        elif self.config.memory_type == "vector":
            self.memory = VectorMemory(max_size=self.config.memory_size)
        else:
            self.logger.warning(f"Unknown memory type: {self.config.memory_type}")
            self.memory = ConversationMemory(max_size=self.config.memory_size)
    
    def _initialize_tools(self):
        """Initialize tools based on config."""
        from .tools import ToolRegistry
        
        registry = ToolRegistry()
        for tool_name in self.config.tools:
            tool = registry.get_tool(tool_name)
            if tool:
                self.tools[tool_name] = tool
                self.logger.info(f"🔧 Loaded tool: {tool_name}")
            else:
                self.logger.warning(f"Tool not found: {tool_name}")
    
    def add_message(self, message: BaseMessage):
        """Add a message to conversation history and memory."""
        self.conversation_history.append(message)
        if self.memory:
            self.memory.add_message(message)
    
    def get_context(self) -> List[BaseMessage]:
        """Get conversation context from memory."""
        context = [SystemMessage(content=self.config.system_prompt)]
        
        if self.memory:
            context.extend(self.memory.get_relevant_messages())
        else:
            # Fallback to recent conversation history
            context.extend(self.conversation_history[-self.config.memory_size:])
        
        return context
    
    @abstractmethod
    def process_message(self, message: str) -> str:
        """
        Process an incoming message and return a response.
        
        Args:
            message: User message to process
            
        Returns:
            Agent's response
        """
        pass
    
    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Use a specific tool.
        
        Args:
            tool_name: Name of the tool to use
            **kwargs: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            self.logger.error(f"Tool '{tool_name}' not available")
            return None
        
        try:
            result = self.tools[tool_name].execute(**kwargs)
            self.logger.info(f"🔧 Used tool '{tool_name}' successfully")
            return result
        except Exception as e:
            self.logger.error(f"Tool '{tool_name}' failed: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        return {
            "id": self.id,
            "name": self.config.name,
            "created_at": self.created_at.isoformat(),
            "model": self.config.model_name,
            "tools": list(self.tools.keys()),
            "memory_type": self.config.memory_type,
            "conversation_length": len(self.conversation_history),
            "status": "active" if self.llm else "inactive"
        }


class SimpleAgent(BaseAgent):
    """
    A simple conversational agent.
    
    Good starting point for learning agent development.
    """
    
    def process_message(self, message: str) -> str:
        """Process a message and return a response."""
        if not self.llm:
            return "❌ Language model not available"
        
        # Add user message to context
        user_message = HumanMessage(content=message)
        self.add_message(user_message)
        
        # Get conversation context
        context = self.get_context()
        
        try:
            # Generate response
            response = self.llm(context)
            
            # Add AI response to context
            ai_message = AIMessage(content=response.content)
            self.add_message(ai_message)
            
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"❌ Sorry, I encountered an error: {str(e)}"


class ToolAgent(BaseAgent):
    """
    An agent that can use tools to perform tasks.
    
    Demonstrates tool integration and decision making.
    """
    
    def process_message(self, message: str) -> str:
        """Process a message, potentially using tools."""
        if not self.llm:
            return "❌ Language model not available"
        
        # Add user message
        user_message = HumanMessage(content=message)
        self.add_message(user_message)
        
        # Analyze if tools are needed
        tool_analysis = self._analyze_tool_needs(message)
        
        if tool_analysis.get("needs_tool") and tool_analysis.get("tool_name"):
            # Use the identified tool
            tool_result = self.use_tool(
                tool_analysis["tool_name"], 
                **tool_analysis.get("tool_args", {})
            )
            
            if tool_result:
                # Include tool result in response generation
                enhanced_message = f"{message}\n\nTool result: {tool_result}"
                user_message = HumanMessage(content=enhanced_message)
        
        # Generate response with context
        context = self.get_context()
        
        try:
            response = self.llm(context)
            ai_message = AIMessage(content=response.content)
            self.add_message(ai_message)
            
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"❌ Sorry, I encountered an error: {str(e)}"
    
    def _analyze_tool_needs(self, message: str) -> Dict[str, Any]:
        """
        Analyze if the message requires tool usage.
        
        This is a simplified analysis. In practice, you might use
        the LLM itself to make this decision.
        """
        message_lower = message.lower()
        
        # Simple keyword-based analysis
        if "calculate" in message_lower or "math" in message_lower:
            return {
                "needs_tool": True,
                "tool_name": "calculator",
                "tool_args": {"expression": message}
            }
        elif "search" in message_lower or "find" in message_lower:
            return {
                "needs_tool": True,
                "tool_name": "search",
                "tool_args": {"query": message}
            }
        elif "time" in message_lower or "date" in message_lower:
            return {
                "needs_tool": True,
                "tool_name": "datetime",
                "tool_args": {}
            }
        
        return {"needs_tool": False}


class AgentManager:
    """
    Manages multiple agents and their interactions.
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger("AgentManager")
    
    def create_agent(self, config: AgentConfig, agent_type: str = "simple") -> BaseAgent:
        """
        Create a new agent.
        
        Args:
            config: Agent configuration
            agent_type: Type of agent to create
            
        Returns:
            Created agent instance
        """
        if agent_type == "simple":
            agent = SimpleAgent(config)
        elif agent_type == "tool":
            agent = ToolAgent(config)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        self.agents[agent.id] = agent
        self.logger.info(f"✅ Created {agent_type} agent: {config.name}")
        
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents and their status."""
        return [agent.get_status() for agent in self.agents.values()]
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent."""
        if agent_id in self.agents:
            agent_name = self.agents[agent_id].config.name
            del self.agents[agent_id]
            self.logger.info(f"🗑️ Removed agent: {agent_name}")
            return True
        return False
    
    def broadcast_message(self, message: str) -> Dict[str, str]:
        """
        Send a message to all agents.
        
        Args:
            message: Message to broadcast
            
        Returns:
            Dictionary of agent responses
        """
        responses = {}
        for agent_id, agent in self.agents.items():
            try:
                response = agent.process_message(message)
                responses[agent.config.name] = response
            except Exception as e:
                responses[agent.config.name] = f"Error: {str(e)}"
        
        return responses


# Example usage
def demo_agents():
    """Demonstrate the agent framework."""
    print("🤖 Agent Framework Demo")
    print("=" * 50)
    
    # Create agent manager
    manager = AgentManager()
    
    # Create a simple agent
    simple_config = AgentConfig(
        name="Assistant",
        description="A helpful assistant",
        system_prompt="You are a friendly and helpful AI assistant.",
        verbose=True
    )
    
    simple_agent = manager.create_agent(simple_config, "simple")
    
    # Create a tool agent
    tool_config = AgentConfig(
        name="ToolBot",
        description="An agent with tools",
        tools=["calculator", "datetime"],
        system_prompt="You are an AI assistant with access to tools. Use them when needed.",
        verbose=True
    )
    
    tool_agent = manager.create_agent(tool_config, "tool")
    
    # Demo conversation
    print("\n📝 Testing Simple Agent:")
    response = simple_agent.process_message("Hello! How are you?")
    print(f"Response: {response}")
    
    print(f"\n🔧 Testing Tool Agent:")
    response = tool_agent.process_message("What time is it?")
    print(f"Response: {response}")
    
    # List all agents
    print(f"\n📊 Agent Status:")
    for status in manager.list_agents():
        print(f"- {status['name']}: {status['status']}")


if __name__ == "__main__":
    demo_agents()