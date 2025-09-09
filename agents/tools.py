"""
Agent Tools

Collection of tools that agents can use to perform various tasks.
"""

import json
import requests
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
import math
import re


class BaseTool(ABC):
    """Base class for all agent tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        pass
    
    def get_info(self) -> Dict[str, str]:
        """Get tool information."""
        return {
            "name": self.name,
            "description": self.description
        }


class CalculatorTool(BaseTool):
    """A calculator tool for mathematical operations."""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs mathematical calculations. Supports basic operations, trigonometry, and more."
        )
    
    def execute(self, expression: str = "", **kwargs) -> str:
        """
        Execute a mathematical expression.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Result of the calculation
        """
        if not expression:
            return "❌ No expression provided"
        
        try:
            # Clean the expression
            expression = expression.strip()
            
            # Simple safety check - only allow certain characters
            allowed_chars = set('0123456789+-*/().sincoxtanlogsqrt ')
            if not all(c.lower() in allowed_chars for c in expression):
                return "❌ Invalid characters in expression"
            
            # Replace common math functions
            expression = expression.replace('sin', 'math.sin')
            expression = expression.replace('cos', 'math.cos') 
            expression = expression.replace('tan', 'math.tan')
            expression = expression.replace('log', 'math.log')
            expression = expression.replace('sqrt', 'math.sqrt')
            
            # Evaluate safely
            result = eval(expression, {"__builtins__": {}, "math": math})
            
            return f"📊 {expression} = {result}"
            
        except Exception as e:
            return f"❌ Calculation error: {str(e)}"


class DateTimeTool(BaseTool):
    """A tool for date and time operations."""
    
    def __init__(self):
        super().__init__(
            name="datetime",
            description="Provides current date and time information."
        )
    
    def execute(self, format_type: str = "default", **kwargs) -> str:
        """
        Get current date and time.
        
        Args:
            format_type: Format for the output (default, iso, custom)
            
        Returns:
            Formatted date and time
        """
        now = datetime.now()
        
        if format_type == "iso":
            return f"🕐 Current time (ISO): {now.isoformat()}"
        elif format_type == "custom":
            custom_format = kwargs.get("custom_format", "%Y-%m-%d %H:%M:%S")
            return f"🕐 Current time: {now.strftime(custom_format)}"
        else:
            return f"🕐 Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


class SearchTool(BaseTool):
    """A web search tool (mock implementation)."""
    
    def __init__(self):
        super().__init__(
            name="search",
            description="Searches the web for information (mock implementation)."
        )
    
    def execute(self, query: str = "", limit: int = 3, **kwargs) -> str:
        """
        Search for information.
        
        Args:
            query: Search query
            limit: Number of results to return
            
        Returns:
            Search results (mock)
        """
        if not query:
            return "❌ No search query provided"
        
        # Mock search results
        mock_results = [
            {
                "title": f"Information about {query}",
                "snippet": f"This is a mock search result for '{query}'. In a real implementation, this would connect to a search API.",
                "url": f"https://example.com/search?q={query.replace(' ', '+')}"
            },
            {
                "title": f"More details on {query}",
                "snippet": f"Additional information about {query} from various sources.",
                "url": f"https://example2.com/info/{query.replace(' ', '-')}"
            },
            {
                "title": f"{query} - Wikipedia",
                "snippet": f"Wikipedia article about {query} with comprehensive information.",
                "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
            }
        ]
        
        # Format results
        results_text = f"🔍 Search results for '{query}':\n\n"
        for i, result in enumerate(mock_results[:limit], 1):
            results_text += f"{i}. **{result['title']}**\n"
            results_text += f"   {result['snippet']}\n"
            results_text += f"   URL: {result['url']}\n\n"
        
        return results_text


class FileReaderTool(BaseTool):
    """A tool for reading files."""
    
    def __init__(self):
        super().__init__(
            name="file_reader",
            description="Reads content from text files."
        )
    
    def execute(self, file_path: str = "", max_chars: int = 1000, **kwargs) -> str:
        """
        Read content from a file.
        
        Args:
            file_path: Path to the file to read
            max_chars: Maximum number of characters to read
            
        Returns:
            File content or error message
        """
        if not file_path:
            return "❌ No file path provided"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read(max_chars)
                
                if len(content) == max_chars:
                    content += "\n... (truncated)"
                
                return f"📄 Content of {file_path}:\n\n{content}"
                
        except FileNotFoundError:
            return f"❌ File not found: {file_path}"
        except PermissionError:
            return f"❌ Permission denied: {file_path}"
        except Exception as e:
            return f"❌ Error reading file: {str(e)}"


class WeatherTool(BaseTool):
    """A weather information tool (mock implementation)."""
    
    def __init__(self):
        super().__init__(
            name="weather",
            description="Gets weather information for a location (mock implementation)."
        )
    
    def execute(self, location: str = "", **kwargs) -> str:
        """
        Get weather information.
        
        Args:
            location: Location to get weather for
            
        Returns:
            Weather information (mock)
        """
        if not location:
            return "❌ No location provided"
        
        # Mock weather data
        import random
        temperatures = [18, 22, 25, 28, 15, 12, 30]
        conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy", "Foggy"]
        
        temp = random.choice(temperatures)
        condition = random.choice(conditions)
        humidity = random.randint(30, 80)
        
        return f"🌤️ Weather in {location}:\n" \
               f"Temperature: {temp}°C\n" \
               f"Condition: {condition}\n" \
               f"Humidity: {humidity}%\n" \
               f"(This is mock data for demonstration)"


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools."""
        default_tools = [
            CalculatorTool(),
            DateTimeTool(),
            SearchTool(),
            FileReaderTool(),
            WeatherTool()
        ]
        
        for tool in default_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool: BaseTool):
        """Register a new tool."""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tools."""
        return [tool.get_info() for tool in self.tools.values()]
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the registry."""
        if name in self.tools:
            del self.tools[name]
            return True
        return False


# Custom tool example
class CustomAPITool(BaseTool):
    """Example of a custom tool that calls an external API."""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        super().__init__(
            name="custom_api",
            description="Calls a custom API endpoint."
        )
        self.api_url = api_url
        self.api_key = api_key
    
    def execute(self, endpoint: str = "", params: Dict = None, **kwargs) -> str:
        """
        Call the custom API.
        
        Args:
            endpoint: API endpoint to call
            params: Parameters to send
            
        Returns:
            API response
        """
        if not endpoint:
            return "❌ No endpoint specified"
        
        try:
            url = f"{self.api_url}/{endpoint}"
            headers = {}
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.get(url, params=params or {}, headers=headers)
            response.raise_for_status()
            
            return f"✅ API Response:\n{response.text}"
            
        except requests.RequestException as e:
            return f"❌ API Error: {str(e)}"


def demo_tools():
    """Demonstrate the tools system."""
    print("🔧 Tools Demo")
    print("=" * 50)
    
    # Create tool registry
    registry = ToolRegistry()
    
    # List available tools
    print("📋 Available Tools:")
    for tool_info in registry.list_tools():
        print(f"- {tool_info['name']}: {tool_info['description']}")
    
    print("\n🧮 Calculator Tool Demo:")
    calc = registry.get_tool("calculator")
    print(calc.execute(expression="2 + 3 * 4"))
    print(calc.execute(expression="sqrt(16) + sin(0)"))
    
    print("\n🕐 DateTime Tool Demo:")
    dt = registry.get_tool("datetime")
    print(dt.execute())
    print(dt.execute(format_type="iso"))
    
    print("\n🔍 Search Tool Demo:")
    search = registry.get_tool("search")
    print(search.execute(query="artificial intelligence", limit=2))
    
    print("\n🌤️ Weather Tool Demo:")
    weather = registry.get_tool("weather")
    print(weather.execute(location="New York"))


if __name__ == "__main__":
    demo_tools()