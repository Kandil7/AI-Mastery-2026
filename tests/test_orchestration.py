"""
Unit Tests for Orchestration Module

Tests for Chain, Agent, Tool, and Memory components.
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.orchestration.orchestration import (
    Chain,
    SequentialChain,
    ParallelChain,
    LambdaChain,
    ConditionalChain,
    Tool,
    ToolRegistry,
    ToolResult,
    CalculatorTool,
    SearchTool,
    Agent,
    ReActAgent,
    Memory,
    ConversationMemory,
    BufferMemory,
    Message,
    MessageRole,
    OrchestrationConfig,
)


class TestLambdaChain(unittest.TestCase):
    """Tests for LambdaChain."""
    
    def test_simple_transform(self):
        """Test simple transformation."""
        chain = LambdaChain(lambda x: x.upper())
        result = chain.run("hello")
        self.assertEqual(result, "HELLO")
    
    def test_numeric_transform(self):
        """Test numeric transformation."""
        chain = LambdaChain(lambda x: x * 2)
        result = chain.run(5)
        self.assertEqual(result, 10)
    
    def test_chain_with_name(self):
        """Test chain with name."""
        chain = LambdaChain(lambda x: x, name="identity")
        self.assertEqual(chain.name, "identity")


class TestSequentialChain(unittest.TestCase):
    """Tests for SequentialChain."""
    
    def test_sequential_execution(self):
        """Test chains execute in sequence."""
        chain = SequentialChain([
            LambdaChain(lambda x: x + 1),
            LambdaChain(lambda x: x * 2),
        ])
        
        result = chain.run(5)
        self.assertEqual(result, 12)  # (5+1)*2 = 12
    
    def test_string_processing(self):
        """Test string processing pipeline."""
        chain = SequentialChain([
            LambdaChain(lambda x: x.strip()),
            LambdaChain(lambda x: x.lower()),
            LambdaChain(lambda x: x.replace(" ", "_")),
        ])
        
        result = chain.run("  Hello World  ")
        self.assertEqual(result, "hello_world")
    
    def test_empty_chain(self):
        """Test empty sequential chain."""
        chain = SequentialChain([])
        result = chain.run("input")
        self.assertEqual(result, "input")


class TestParallelChain(unittest.TestCase):
    """Tests for ParallelChain."""
    
    def test_parallel_execution(self):
        """Test chains execute in parallel."""
        chain = ParallelChain([
            LambdaChain(lambda x: x + 1),
            LambdaChain(lambda x: x * 2),
            LambdaChain(lambda x: x ** 2),
        ])
        
        results = chain.run(3)
        self.assertEqual(results, [4, 6, 9])
    
    def test_single_chain(self):
        """Test parallel chain with single chain."""
        chain = ParallelChain([LambdaChain(lambda x: x.upper())])
        results = chain.run("hello")
        self.assertEqual(results, ["HELLO"])


class TestConditionalChain(unittest.TestCase):
    """Tests for ConditionalChain."""
    
    def test_condition_true(self):
        """Test when condition is true."""
        chain = ConditionalChain(
            condition=lambda x: x > 0,
            if_true=LambdaChain(lambda x: "positive"),
            if_false=LambdaChain(lambda x: "non-positive")
        )
        
        result = chain.run(5)
        self.assertEqual(result, "positive")
    
    def test_condition_false(self):
        """Test when condition is false."""
        chain = ConditionalChain(
            condition=lambda x: x > 0,
            if_true=LambdaChain(lambda x: "positive"),
            if_false=LambdaChain(lambda x: "non-positive")
        )
        
        result = chain.run(-3)
        self.assertEqual(result, "non-positive")


class TestCalculatorTool(unittest.TestCase):
    """Tests for CalculatorTool."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = CalculatorTool()
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic."""
        result = self.calculator.run(expression="2 + 3")
        self.assertTrue(result.success)
        self.assertEqual(result.output, 5)
    
    def test_multiplication(self):
        """Test multiplication."""
        result = self.calculator.run(expression="4 * 5")
        self.assertTrue(result.success)
        self.assertEqual(result.output, 20)
    
    def test_division(self):
        """Test division."""
        result = self.calculator.run(expression="10 / 2")
        self.assertTrue(result.success)
        self.assertEqual(result.output, 5.0)
    
    def test_complex_expression(self):
        """Test complex expression."""
        result = self.calculator.run(expression="(2 + 3) * 4")
        self.assertTrue(result.success)
        self.assertEqual(result.output, 20)
    
    def test_invalid_expression(self):
        """Test invalid expression."""
        result = self.calculator.run(expression="invalid")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)


class TestSearchTool(unittest.TestCase):
    """Tests for SearchTool."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.search = SearchTool()
    
    def test_search_returns_result(self):
        """Test search returns results."""
        result = self.search.run(query="machine learning")
        self.assertTrue(result.success)
        self.assertIsNotNone(result.output)
    
    def test_empty_query(self):
        """Test empty query."""
        result = self.search.run(query="")
        # Should handle gracefully
        self.assertIsInstance(result, ToolResult)


class TestToolRegistry(unittest.TestCase):
    """Tests for ToolRegistry."""
    
    def test_register_and_get(self):
        """Test registering and getting tools."""
        registry = ToolRegistry()
        calc = CalculatorTool()
        
        registry.register(calc)
        retrieved = registry.get("calculator")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "calculator")
    
    def test_list_tools(self):
        """Test listing tools."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(SearchTool())
        
        tools = registry.list_tools()
        self.assertEqual(len(tools), 2)
    
    def test_get_nonexistent(self):
        """Test getting non-existent tool."""
        registry = ToolRegistry()
        tool = registry.get("nonexistent")
        self.assertIsNone(tool)
    
    def test_to_prompt(self):
        """Test generating prompt description."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        
        prompt = registry.to_prompt()
        self.assertIn("calculator", prompt)


class TestMessage(unittest.TestCase):
    """Tests for Message dataclass."""
    
    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(role=MessageRole.USER, content="Hello")
        
        self.assertEqual(msg.role, MessageRole.USER)
        self.assertEqual(msg.content, "Hello")
    
    def test_message_roles(self):
        """Test different message roles."""
        for role in [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.SYSTEM]:
            msg = Message(role=role, content="Test")
            self.assertEqual(msg.role, role)


class TestConversationMemory(unittest.TestCase):
    """Tests for ConversationMemory."""
    
    def test_add_and_retrieve(self):
        """Test adding and retrieving messages."""
        memory = ConversationMemory(max_messages=10)
        
        memory.add_message(Message(role=MessageRole.USER, content="Hello"))
        memory.add_message(Message(role=MessageRole.ASSISTANT, content="Hi there"))
        
        self.assertEqual(len(memory.messages), 2)
    
    def test_max_messages_limit(self):
        """Test max messages limit."""
        memory = ConversationMemory(max_messages=3)
        
        for i in range(5):
            memory.add_message(Message(role=MessageRole.USER, content=f"Msg {i}"))
        
        self.assertEqual(len(memory.messages), 3)
    
    def test_to_context(self):
        """Test converting to context string."""
        memory = ConversationMemory()
        memory.add_message(Message(role=MessageRole.USER, content="Question"))
        memory.add_message(Message(role=MessageRole.ASSISTANT, content="Answer"))
        
        context = memory.to_context()
        
        self.assertIn("Question", context)
        self.assertIn("Answer", context)
    
    def test_clear(self):
        """Test clearing memory."""
        memory = ConversationMemory()
        memory.add_message(Message(role=MessageRole.USER, content="Test"))
        memory.clear()
        
        self.assertEqual(len(memory.messages), 0)


class TestBufferMemory(unittest.TestCase):
    """Tests for BufferMemory."""
    
    def test_add_and_get(self):
        """Test adding and getting values."""
        memory = BufferMemory()
        memory.add("key1", "value1")
        
        self.assertEqual(memory.get("key1"), "value1")
    
    def test_get_nonexistent(self):
        """Test getting non-existent key."""
        memory = BufferMemory()
        result = memory.get("nonexistent")
        self.assertIsNone(result)
    
    def test_get_all(self):
        """Test getting all values."""
        memory = BufferMemory()
        memory.add("a", 1)
        memory.add("b", 2)
        
        all_values = memory.get_all()
        self.assertEqual(len(all_values), 2)


class TestReActAgent(unittest.TestCase):
    """Tests for ReActAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()
        self.registry.register(CalculatorTool())
        self.agent = ReActAgent(tools=self.registry, max_iterations=5)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.max_iterations, 5)
    
    def test_react_prompt_template(self):
        """Test ReAct prompt template exists."""
        self.assertIsNotNone(self.agent.REACT_PROMPT)
        self.assertTrue(len(self.agent.REACT_PROMPT) > 0)


class TestOrchestrationConfig(unittest.TestCase):
    """Tests for OrchestrationConfig."""
    
    def test_default_values(self):
        """Test default configuration."""
        config = OrchestrationConfig()
        
        self.assertEqual(config.max_iterations, 10)
        self.assertEqual(config.timeout, 30)
        self.assertTrue(config.verbose)


if __name__ == "__main__":
    unittest.main()
