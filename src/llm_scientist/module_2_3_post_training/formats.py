"""
Data Formats - Module 2.3.1

Production-ready implementations of common LLM data formats:
- ShareGPT format
- ChatML format
- Alpaca format
- Conversation templates
- Format conversion utilities

References:
- ShareGPT: https://sharegpt.com/
- ChatML: https://github.com/openai/openai-python/blob/main/chatml.md
- Alpaca: https://github.com/tatsu-lab/stanford_alpaca
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Message roles in conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in a conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'role': self.role.value if isinstance(self.role, MessageRole) else self.role,
            'content': self.content,
        }
        if self.name:
            result['name'] = self.name
        if self.function_call:
            result['function_call'] = self.function_call
        if self.tool_calls:
            result['tool_calls'] = self.tool_calls
        if self.tool_call_id:
            result['tool_call_id'] = self.tool_call_id
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create from dictionary."""
        role = data.get('role', 'user')
        if isinstance(role, str):
            role = MessageRole(role)
        
        return cls(
            role=role,
            content=data.get('content', ''),
            name=data.get('name'),
            function_call=data.get('function_call'),
            tool_calls=data.get('tool_calls'),
            tool_call_id=data.get('tool_call_id'),
        )


@dataclass
class Conversation:
    """A conversation with multiple messages."""
    messages: List[Message]
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(
        self,
        role: Union[str, MessageRole],
        content: str,
        **kwargs,
    ) -> None:
        """Add a message to the conversation."""
        if isinstance(role, str):
            role = MessageRole(role)
        self.messages.append(Message(role=role, content=content, **kwargs))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'messages': [m.to_dict() for m in self.messages],
            'system_prompt': self.system_prompt,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create from dictionary."""
        messages = [Message.from_dict(m) for m in data.get('messages', [])]
        return cls(
            messages=messages,
            system_prompt=data.get('system_prompt'),
            metadata=data.get('metadata', {}),
        )


class ConversationTemplate:
    """
    Conversation Template for formatting conversations.
    
    Defines how conversations should be formatted for different models.
    
    Args:
        name: Template name
        system_template: Template for system message
        user_template: Template for user message
        assistant_template: Template for assistant message
        sep: Separator between messages
        sep2: Second separator (for some formats)
        stop_tokens: List of stop tokens
        
    Example:
        >>> template = ConversationTemplate(
        ...     name="llama-2",
        ...     system_template="[INST] <<SYS>>\n{content}\n<</SYS>>\n\n",
        ...     user_template="{content} [/INST]",
        ...     assistant_template="{content}",
        ... )
        >>> formatted = template.format(conversation)
    """
    
    def __init__(
        self,
        name: str,
        system_template: str = "{content}",
        user_template: str = "{content}",
        assistant_template: str = "{content}",
        sep: str = "\n",
        sep2: Optional[str] = None,
        stop_tokens: Optional[List[str]] = None,
        add_generation_prompt: bool = False,
    ):
        self.name = name
        self.system_template = system_template
        self.user_template = user_template
        self.assistant_template = assistant_template
        self.sep = sep
        self.sep2 = sep2 or self.sep
        self.stop_tokens = stop_tokens or []
        self.add_generation_prompt = add_generation_prompt
    
    def format(self, conversation: Conversation) -> str:
        """
        Format a conversation using this template.
        
        Args:
            conversation: Conversation to format
        
        Returns:
            Formatted string
        """
        result = []
        
        # Add system message if present
        if conversation.system_prompt:
            result.append(
                self.system_template.format(content=conversation.system_prompt)
            )
        
        # Add messages
        for i, message in enumerate(conversation.messages):
            if message.role == MessageRole.USER:
                result.append(
                    self.user_template.format(content=message.content)
                )
            elif message.role == MessageRole.ASSISTANT:
                result.append(
                    self.assistant_template.format(content=message.content)
                )
            elif message.role == MessageRole.SYSTEM:
                result.append(
                    self.system_template.format(content=message.content)
                )
        
        # Add generation prompt if requested
        if self.add_generation_prompt:
            result.append(self.user_template.format(content=""))
        
        return self.sep.join(result)
    
    @classmethod
    def get_template(cls, name: str) -> 'ConversationTemplate':
        """Get a predefined template by name."""
        templates = {
            'llama-2': cls(
                name='llama-2',
                system_template="[INST] <<SYS>>\n{content}\n<</SYS>>\n\n",
                user_template="{content} [/INST]",
                assistant_template="{content} ",
                sep="",
                stop_tokens=["</s>"],
            ),
            'llama-3': cls(
                name='llama-3',
                system_template="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
                user_template="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
                assistant_template="<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
                sep="",
                stop_tokens=["<|eot_id|>"],
            ),
            'chatml': cls(
                name='chatml',
                system_template="<|im_start|>system\n{content}<|im_end|>",
                user_template="<|im_start|>user\n{content}<|im_end|>",
                assistant_template="<|im_start|>assistant\n{content}<|im_end|>",
                sep="\n",
                stop_tokens=["<|im_end|>"],
            ),
            'vicuna': cls(
                name='vicuna',
                system_template="A chat between a curious user and an artificial intelligence assistant. {content}",
                user_template="USER: {content}",
                assistant_template="ASSISTANT: {content}",
                sep=" ",
                stop_tokens=["</s>"],
            ),
            'mistral': cls(
                name='mistral',
                system_template="[INST] {content} [/INST]",
                user_template="[INST] {content} [/INST]",
                assistant_template="{content}",
                sep=" ",
                stop_tokens=["</s>"],
            ),
            'zephyr': cls(
                name='zephyr',
                system_template="<|system|>\n{content}</s>",
                user_template="<|user|>\n{content}</s>",
                assistant_template="<|assistant|>\n{content}</s>",
                sep="\n",
                stop_tokens=["</s>"],
            ),
            'alpaca': cls(
                name='alpaca',
                system_template="Below is an instruction that describes a task. {content}\n\n",
                user_template="### Instruction:\n{content}\n\n### Response:",
                assistant_template="{content}",
                sep="\n\n",
                stop_tokens=["###"],
            ),
        }
        
        if name not in templates:
            raise ValueError(f"Unknown template: {name}. Available: {list(templates.keys())}")
        
        return templates[name]


class BaseFormat(ABC):
    """Abstract base class for data formats."""
    
    @abstractmethod
    def parse(self, data: Union[str, Dict, List]) -> List[Conversation]:
        """Parse data into conversations."""
        pass
    
    @abstractmethod
    def serialize(self, conversations: List[Conversation]) -> Union[str, Dict, List]:
        """Serialize conversations to format."""
        pass
    
    def load_file(self, path: Union[str, Path]) -> List[Conversation]:
        """Load conversations from file."""
        path = Path(path)
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix == '.json':
                data = json.load(f)
            elif path.suffix == '.jsonl':
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
        
        return self.parse(data)
    
    def save_file(
        self,
        conversations: List[Conversation],
        path: Union[str, Path],
    ) -> None:
        """Save conversations to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.serialize(conversations)
        
        with open(path, 'w', encoding='utf-8') as f:
            if path.suffix == '.jsonl':
                for item in data if isinstance(data, list) else [data]:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                json.dump(data, f, ensure_ascii=False, indent=2)


class ShareGPTFormat(BaseFormat):
    """
    ShareGPT Format Parser/Serializer.
    
    ShareGPT format is commonly used for chat datasets.
    
    Format:
    {
        "id": "conversation_id",
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there!"}
        ]
    }
    
    Example:
        >>> format = ShareGPTFormat()
        >>> conversations = format.load_file('data.jsonl')
    """
    
    def __init__(
        self,
        human_role: str = "human",
        gpt_role: str = "gpt",
        system_role: str = "system",
    ):
        self.human_role = human_role
        self.gpt_role = gpt_role
        self.system_role = system_role
    
    def _map_role(self, role: str) -> MessageRole:
        """Map ShareGPT role to MessageRole."""
        role = role.lower()
        
        if role == self.human_role.lower():
            return MessageRole.USER
        elif role == self.gpt_role.lower():
            return MessageRole.ASSISTANT
        elif role == self.system_role.lower():
            return MessageRole.SYSTEM
        else:
            # Try to parse as MessageRole
            try:
                return MessageRole(role)
            except ValueError:
                return MessageRole.USER
    
    def parse(self, data: Union[str, Dict, List]) -> List[Conversation]:
        """Parse ShareGPT format data."""
        if isinstance(data, str):
            data = json.loads(data)
        
        if isinstance(data, dict):
            data = [data]
        
        conversations = []
        
        for item in data:
            conv_data = item.get('conversations', [])
            
            # Extract system prompt if present
            system_prompt = None
            messages = []
            
            for msg in conv_data:
                role = self._map_role(msg.get('from', 'human'))
                content = msg.get('value', '')
                
                if role == MessageRole.SYSTEM:
                    system_prompt = content
                else:
                    messages.append(Message(role=role, content=content))
            
            conversations.append(Conversation(
                messages=messages,
                system_prompt=system_prompt,
                metadata={
                    'id': item.get('id'),
                    'source': 'sharegpt',
                },
            ))
        
        return conversations
    
    def serialize(self, conversations: List[Conversation]) -> List[Dict]:
        """Serialize conversations to ShareGPT format."""
        result = []
        
        for i, conv in enumerate(conversations):
            item = {
                'id': conv.metadata.get('id', f'conv_{i}'),
                'conversations': [],
            }
            
            # Add system prompt if present
            if conv.system_prompt:
                item['conversations'].append({
                    'from': self.system_role,
                    'value': conv.system_prompt,
                })
            
            # Add messages
            for msg in conv.messages:
                if msg.role == MessageRole.USER:
                    from_role = self.human_role
                elif msg.role == MessageRole.ASSISTANT:
                    from_role = self.gpt_role
                elif msg.role == MessageRole.SYSTEM:
                    from_role = self.system_role
                else:
                    from_role = msg.role.value
                
                item['conversations'].append({
                    'from': from_role,
                    'value': msg.content,
                })
            
            result.append(item)
        
        return result


class ChatMLFormat(BaseFormat):
    """
    ChatML Format Parser/Serializer.
    
    ChatML is OpenAI's format for chat models.
    
    Format:
    [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    Example:
        >>> format = ChatMLFormat()
        >>> conversations = format.load_file('data.jsonl')
    """
    
    def parse(self, data: Union[str, Dict, List]) -> List[Conversation]:
        """Parse ChatML format data."""
        if isinstance(data, str):
            data = json.loads(data)
        
        if isinstance(data, dict):
            # Single conversation with messages key
            data = [data]
        
        conversations = []
        
        for item in data:
            messages_data = item.get('messages', item.get('conversations', []))
            
            messages = []
            system_prompt = None
            
            for msg in messages_data:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role == 'system':
                    system_prompt = content
                else:
                    messages.append(Message(
                        role=MessageRole(role),
                        content=content,
                        name=msg.get('name'),
                        function_call=msg.get('function_call'),
                        tool_calls=msg.get('tool_calls'),
                        tool_call_id=msg.get('tool_call_id'),
                    ))
            
            conversations.append(Conversation(
                messages=messages,
                system_prompt=system_prompt,
                metadata={
                    'id': item.get('id'),
                    'source': 'chatml',
                },
            ))
        
        return conversations
    
    def serialize(self, conversations: List[Conversation]) -> List[Dict]:
        """Serialize conversations to ChatML format."""
        result = []
        
        for conv in conversations:
            item = {
                'messages': [],
                **{k: v for k, v in conv.metadata.items() if k != 'source'},
            }
            
            # Add system prompt if present
            if conv.system_prompt:
                item['messages'].append({
                    'role': 'system',
                    'content': conv.system_prompt,
                })
            
            # Add messages
            for msg in conv.messages:
                item['messages'].append(msg.to_dict())
            
            result.append(item)
        
        return result


class AlpacaFormat(BaseFormat):
    """
    Alpaca Format Parser/Serializer.
    
    Alpaca format is used for instruction-following datasets.
    
    Format:
    {
        "instruction": "Translate to French",
        "input": "Hello, how are you?",
        "output": "Bonjour, comment allez-vous?"
    }
    
    Example:
        >>> format = AlpacaFormat()
        >>> conversations = format.load_file('data.json')
    """
    
    def parse(self, data: Union[str, Dict, List]) -> List[Conversation]:
        """Parse Alpaca format data."""
        if isinstance(data, str):
            data = json.loads(data)
        
        if isinstance(data, dict):
            data = [data]
        
        conversations = []
        
        for item in data:
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            output = item.get('output', '')
            
            # Build user content
            if input_text:
                user_content = f"{instruction}\n\n{input_text}"
            else:
                user_content = instruction
            
            # Handle system prompt
            system_prompt = item.get('system')
            
            messages = [
                Message(role=MessageRole.USER, content=user_content),
                Message(role=MessageRole.ASSISTANT, content=output),
            ]
            
            conversations.append(Conversation(
                messages=messages,
                system_prompt=system_prompt,
                metadata={
                    'id': item.get('id'),
                    'source': 'alpaca',
                    'category': item.get('category'),
                },
            ))
        
        return conversations
    
    def serialize(self, conversations: List[Conversation]) -> List[Dict]:
        """Serialize conversations to Alpaca format."""
        result = []
        
        for conv in conversations:
            # Extract instruction and input from first user message
            user_msg = next(
                (m for m in conv.messages if m.role == MessageRole.USER),
                None,
            )
            
            if user_msg is None:
                continue
            
            content = user_msg.content
            if '\n\n' in content:
                parts = content.split('\n\n', 1)
                instruction = parts[0]
                input_text = parts[1] if len(parts) > 1 else ''
            else:
                instruction = content
                input_text = ''
            
            # Extract output from first assistant message
            assistant_msg = next(
                (m for m in conv.messages if m.role == MessageRole.ASSISTANT),
                None,
            )
            output = assistant_msg.content if assistant_msg else ''
            
            item = {
                'instruction': instruction,
                'input': input_text,
                'output': output,
            }
            
            if conv.system_prompt:
                item['system'] = conv.system_prompt
            
            result.append(item)
        
        return result


class FormatConverter:
    """
    Format Converter for converting between formats.
    
    Example:
        >>> converter = FormatConverter()
        >>> conversations = converter.convert(
        ...     'sharegpt', 'chatml', input_path='data.jsonl', output_path='output.jsonl'
        ... )
    """
    
    def __init__(self):
        self._formats = {
            'sharegpt': ShareGPTFormat,
            'chatml': ChatMLFormat,
            'alpaca': AlpacaFormat,
        }
    
    def register_format(self, name: str, format_cls: type) -> None:
        """Register a new format."""
        self._formats[name] = format_cls
    
    def get_format(self, name: str) -> BaseFormat:
        """Get a format by name."""
        if name not in self._formats:
            raise ValueError(f"Unknown format: {name}. Available: {list(self._formats.keys())}")
        return self._formats[name]()
    
    def convert(
        self,
        from_format: str,
        to_format: str,
        input_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        data: Optional[Union[str, Dict, List]] = None,
    ) -> List[Conversation]:
        """
        Convert data between formats.
        
        Args:
            from_format: Source format name
            to_format: Target format name
            input_path: Path to input file
            output_path: Path to output file
            data: Direct data input (alternative to file)
        
        Returns:
            Converted conversations
        """
        # Get formats
        source_format = self.get_format(from_format)
        target_format = self.get_format(to_format)
        
        # Parse source
        if data is not None:
            conversations = source_format.parse(data)
        elif input_path is not None:
            conversations = source_format.load_file(input_path)
        else:
            raise ValueError("Either input_path or data must be provided")
        
        # Save target
        if output_path is not None:
            target_format.save_file(conversations, output_path)
        
        return conversations
    
    def to_template(
        self,
        conversations: List[Conversation],
        template_name: str,
    ) -> List[str]:
        """
        Convert conversations to formatted strings using a template.
        
        Args:
            conversations: Conversations to format
            template_name: Template name
        
        Returns:
            List of formatted strings
        """
        template = ConversationTemplate.get_template(template_name)
        return [template.format(conv) for conv in conversations]


def create_training_example(
    conversation: Conversation,
    template: Optional[ConversationTemplate] = None,
    template_name: str = 'chatml',
    add_generation_prompt: bool = False,
) -> Dict[str, str]:
    """
    Create a training example from a conversation.
    
    Args:
        conversation: Conversation to convert
        template: Optional template to use
        template_name: Template name if template not provided
        add_generation_prompt: Whether to add generation prompt
    
    Returns:
        Dictionary with 'text' (formatted) and 'messages' (structured)
    """
    if template is None:
        template = ConversationTemplate.get_template(template_name)
    
    # Create a copy with add_generation_prompt setting
    template.add_generation_prompt = add_generation_prompt
    
    formatted = template.format(conversation)
    
    return {
        'text': formatted,
        'messages': [m.to_dict() for m in conversation.messages],
        'system_prompt': conversation.system_prompt,
    }


def batch_create_examples(
    conversations: List[Conversation],
    template_name: str = 'chatml',
    add_generation_prompt: bool = False,
) -> List[Dict[str, str]]:
    """
    Create training examples from multiple conversations.
    
    Args:
        conversations: List of conversations
        template_name: Template name
        add_generation_prompt: Whether to add generation prompt
    
    Returns:
        List of training examples
    """
    return [
        create_training_example(conv, template_name=template_name, add_generation_prompt=add_generation_prompt)
        for conv in conversations
    ]
