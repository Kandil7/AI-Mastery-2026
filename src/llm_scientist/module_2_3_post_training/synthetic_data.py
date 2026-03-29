"""
Synthetic Data Generation - Module 2.3.2

Production-ready synthetic data generation:
- Instruction generation
- Seed task generation
- Self-Instruct pipeline
- Quality filtering

References:
- "Self-Instruct: Aligning Language Models with Self-Generated Instructions" (Wang et al., 2022)
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from torch.utils.data import Dataset

from .formats import Conversation, Message, MessageRole

logger = logging.getLogger(__name__)


@dataclass
class InstructionExample:
    """A synthetic instruction example."""
    instruction: str
    input: str = ""
    output: str = ""
    category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_conversation(self, system_prompt: Optional[str] = None) -> Conversation:
        """Convert to Conversation format."""
        messages = []
        
        # Combine instruction and input
        if self.input:
            user_content = f"{self.instruction}\n\n{self.input}"
        else:
            user_content = self.instruction
        
        messages.append(Message(role=MessageRole.USER, content=user_content))
        messages.append(Message(role=MessageRole.ASSISTANT, content=self.output))
        
        return Conversation(
            messages=messages,
            system_prompt=system_prompt,
            metadata={
                'category': self.category,
                **self.metadata,
            },
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'instruction': self.instruction,
            'input': self.input,
            'output': self.output,
            'category': self.category,
            'metadata': self.metadata,
        }


class InstructionGenerator:
    """
    Instruction Generator for creating synthetic instructions.
    
    Uses an LLM to generate diverse instructions based on seed tasks
    and categories.
    
    Args:
        model: LLM model for generation
        tokenizer: Tokenizer for the model
        categories: List of instruction categories
        
    Example:
        >>> generator = InstructionGenerator(model, tokenizer)
        >>> instructions = generator.generate(num_instructions=100)
    """
    
    # Default instruction categories
    DEFAULT_CATEGORIES = [
        "brainstorming",
        "classification",
        "closed QA",
        "creative writing",
        "extraction",
        "generation",
        "information seeking",
        "open QA",
        "reasoning",
        "rewriting",
        "summarization",
        "translation",
    ]
    
    # Generation prompt template
    GENERATION_PROMPT = """You are an instruction generator. Generate diverse, high-quality instructions for the given category.

Category: {category}

Generate {num_instructions} unique instructions. Each instruction should:
1. Be clear and specific
2. Require different skills/knowledge
3. Vary in complexity
4. Be answerable without external tools

Instructions:
{existing_instructions}

New instructions (one per line):
"""
    
    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        categories: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.categories = categories or self.DEFAULT_CATEGORIES
        self.seed = seed
        
        random.seed(seed)
        self._generated: Set[str] = set()
    
    def _generate_with_llm(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate text using the LLM."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be provided")
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):].strip()
    
    def _generate_with_template(
        self,
        category: str,
        num_instructions: int,
        existing: List[str],
    ) -> List[str]:
        """Generate instructions for a category."""
        existing_text = "\n".join(f"- {i}" for i in existing[-10:]) if existing else "None yet"
        
        prompt = self.GENERATION_PROMPT.format(
            category=category,
            num_instructions=num_instructions,
            existing_instructions=existing_text,
        )
        
        response = self._generate_with_llm(prompt)
        
        # Parse instructions from response
        instructions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and not line.startswith('-'):
                # Clean up numbering
                line = line.lstrip('1234567890. ')
                if line and line not in self._generated:
                    instructions.append(line)
                    self._generated.add(line)
        
        return instructions
    
    def generate(
        self,
        num_instructions: int = 100,
        category_weights: Optional[Dict[str, float]] = None,
    ) -> List[InstructionExample]:
        """
        Generate synthetic instructions.
        
        Args:
            num_instructions: Number of instructions to generate
            category_weights: Optional weights for categories
        
        Returns:
            List of InstructionExample
        """
        instructions = []
        
        # Determine category distribution
        if category_weights:
            categories = list(category_weights.keys())
            weights = list(category_weights.values())
        else:
            categories = self.categories
            weights = [1.0] * len(categories)
        
        # Generate instructions per category
        per_category = max(1, num_instructions // len(categories))
        
        for category in categories:
            category_instructions = self._generate_with_template(
                category=category,
                num_instructions=per_category,
                existing=[],
            )
            
            for instr in category_instructions[:per_category]:
                instructions.append(InstructionExample(
                    instruction=instr,
                    category=category,
                ))
        
        return instructions[:num_instructions]
    
    def generate_from_seeds(
        self,
        seed_instructions: List[str],
        num_new: int = 50,
    ) -> List[InstructionExample]:
        """
        Generate new instructions based on seed instructions.
        
        Args:
            seed_instructions: List of seed instructions
            num_new: Number of new instructions to generate
        
        Returns:
            List of new InstructionExample
        """
        # Add seeds to generated set
        for instr in seed_instructions:
            self._generated.add(instr)
        
        # Generate variations
        new_instructions = []
        
        prompt = f"""Given these example instructions, generate {num_new} new, diverse instructions that are similar in style but cover different topics:

Examples:
{chr(10).join(f"- {s}" for s in seed_instructions[:10])}

New instructions (one per line):
"""
        
        if self.model and self.tokenizer:
            response = self._generate_with_llm(prompt)
            
            for line in response.split('\n'):
                line = line.strip().lstrip('1234567890. -')
                if line and line not in self._generated:
                    new_instructions.append(InstructionExample(
                        instruction=line,
                        category="mixed",
                    ))
                    self._generated.add(line)
        
        return new_instructions[:num_new]


class SeedTaskGenerator:
    """
    Seed Task Generator for bootstrapping instruction generation.
    
    Generates initial seed tasks that can be used to bootstrap
    the Self-Instruct process.
    
    Example:
        >>> generator = SeedTaskGenerator()
        >>> seeds = generator.generate(num_seeds=175)
    """
    
    # Pre-defined seed task templates
    TASK_TEMPLATES = [
        "Write a {type} about {topic}",
        "Explain {concept} in simple terms",
        "Compare {item1} and {item2}",
        "List {num} {type} of {category}",
        "What is the difference between {item1} and {item2}?",
        "How does {process} work?",
        "Why is {phenomenon} important?",
        "Describe the steps to {action}",
        "What are the benefits of {thing}?",
        "What are the drawbacks of {thing}?",
        "Summarize the following: {text}",
        "Translate '{text}' to {language}",
        "Rewrite the following in a {style} style: {text}",
        "Classify the following as {categories}: {text}",
        "Extract all {entity_type} from: {text}",
        "Generate a {type} with these properties: {properties}",
        "Answer the question: {question}",
        "Solve this problem: {problem}",
        "Debug the following code: {code}",
        "Write code to {task}",
    ]
    
    # Topic categories
    TOPICS = [
        "technology", "science", "history", "literature", "art",
        "music", "sports", "cooking", "travel", "health",
        "business", "education", "politics", "environment", "philosophy",
    ]
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self._generated: Set[str] = set()
    
    def generate(
        self,
        num_seeds: int = 175,
        templates: Optional[List[str]] = None,
    ) -> List[InstructionExample]:
        """
        Generate seed tasks.
        
        Args:
            num_seeds: Number of seed tasks
            templates: Optional custom templates
        
        Returns:
            List of seed InstructionExample
        """
        templates = templates or self.TASK_TEMPLATES
        seeds = []
        
        while len(seeds) < num_seeds:
            template = random.choice(templates)
            
            # Fill in template placeholders
            try:
                filled = self._fill_template(template)
                
                if filled not in self._generated:
                    seeds.append(InstructionExample(
                        instruction=filled,
                        category="seed",
                    ))
                    self._generated.add(filled)
            except Exception:
                continue
        
        return seeds
    
    def _fill_template(self, template: str) -> str:
        """Fill in template placeholders with random values."""
        replacements = {
            'type': random.choice(['story', 'article', 'poem', 'essay', 'report']),
            'topic': random.choice(self.TOPICS),
            'concept': random.choice(['gravity', 'democracy', 'photosynthesis', 'machine learning']),
            'item1': random.choice(['cats', 'dogs', 'Python', 'Java']),
            'item2': random.choice(['dogs', 'cats', 'JavaScript', 'C++']),
            'num': str(random.randint(3, 10)),
            'category': random.choice(self.TOPICS),
            'process': random.choice(['digestion', 'compilation', 'election']),
            'phenomenon': random.choice(['climate change', 'biodiversity', 'innovation']),
            'action': random.choice(['bake a cake', 'write a program', 'plant a tree']),
            'thing': random.choice(['exercise', 'meditation', 'reading']),
            'text': 'A short passage about daily life.',
            'language': random.choice(['French', 'Spanish', 'German', 'Japanese']),
            'style': random.choice(['formal', 'casual', 'humorous', 'professional']),
            'categories': random.choice(['positive/negative', 'fact/opinion']),
            'entity_type': random.choice(['names', 'dates', 'locations']),
            'properties': 'clear structure and engaging content',
            'question': random.choice(['What is AI?', 'How do vaccines work?']),
            'problem': random.choice(['2x + 5 = 15', 'Find the area of a circle']),
            'code': 'def hello(): print("Hello")',
            'task': random.choice(['sort a list', 'fetch API data', 'parse JSON']),
        }
        
        result = template
        for key, value in replacements.items():
            result = result.replace('{' + key + '}', value)
        
        return result


class SelfInstruct:
    """
    Self-Instruct Pipeline.
    
    Implements the full Self-Instruct algorithm for generating
    instruction-following data.
    
    Args:
        model: LLM model
        tokenizer: Tokenizer
        num_instructions: Target number of instructions
        num_seed_tasks: Number of seed tasks
        
    Reference:
        "Self-Instruct: Aligning Language Models with Self-Generated Instructions" (Wang et al., 2022)
    
    Example:
        >>> self_instruct = SelfInstruct(model, tokenizer)
        >>> dataset = self_instruct.run()
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        num_instructions: int = 52000,
        num_seed_tasks: int = 175,
        num_generations_per_seed: int = 5,
        diversity_threshold: float = 0.7,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_instructions = num_instructions
        self.num_seed_tasks = num_seed_tasks
        self.num_generations_per_seed = num_generations_per_seed
        self.diversity_threshold = diversity_threshold
        
        self.seed_generator = SeedTaskGenerator()
        self.instruction_generator = InstructionGenerator(model, tokenizer)
        self._instructions: List[InstructionExample] = []
        self._instruction_set: Set[str] = set()
    
    def _generate_response(
        self,
        instruction: InstructionExample,
    ) -> str:
        """Generate a response for an instruction."""
        if not self.model or not self.tokenizer:
            # Fallback: return placeholder
            return f"Response to: {instruction.instruction}"
        
        prompt = f"""Instruction: {instruction.instruction}

Input: {instruction.input if instruction.input else 'N/A'}

Response:
"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split('Response:')[-1].strip()
    
    def _check_diversity(
        self,
        new_instruction: str,
        min_similarity: float = 0.7,
    ) -> bool:
        """Check if instruction is diverse enough."""
        # Simple diversity check using Jaccard similarity
        new_words = set(new_instruction.lower().split())
        
        for existing in self._instruction_set:
            existing_words = set(existing.lower().split())
            
            if not new_words or not existing_words:
                continue
            
            intersection = len(new_words & existing_words)
            union = len(new_words | existing_words)
            similarity = intersection / union if union > 0 else 0
            
            if similarity > min_similarity:
                return False
        
        return True
    
    def run(self) -> List[InstructionExample]:
        """
        Run the Self-Instruct pipeline.
        
        Returns:
            List of generated InstructionExample
        """
        logger.info("Starting Self-Instruct pipeline...")
        
        # Step 1: Generate seed tasks
        logger.info(f"Generating {self.num_seed_tasks} seed tasks...")
        seed_tasks = self.seed_generator.generate(self.num_seed_tasks)
        self._instructions.extend(seed_tasks)
        self._instruction_set.update(t.instruction for t in seed_tasks)
        
        # Step 2: Iteratively generate new instructions
        iterations = 0
        max_iterations = self.num_instructions // self.num_generations_per_seed
        
        while len(self._instructions) < self.num_instructions and iterations < max_iterations:
            iterations += 1
            
            # Sample seed tasks
            sampled_seeds = random.sample(
                seed_tasks,
                min(self.num_generations_per_seed, len(seed_tasks)),
            )
            
            # Generate new instructions
            for seed in sampled_seeds:
                if len(self._instructions) >= self.num_instructions:
                    break
                
                # Generate variations
                new_instructions = self.instruction_generator.generate_from_seeds(
                    [seed.instruction],
                    num_new=1,
                )
                
                for new_instr in new_instructions:
                    if self._check_diversity(new_instr.instruction):
                        # Generate response
                        new_instr.output = self._generate_response(new_instr)
                        
                        self._instructions.append(new_instr)
                        self._instruction_set.add(new_instr.instruction)
            
            logger.info(f"Iteration {iterations}: {len(self._instructions)} instructions")
        
        logger.info(f"Generated {len(self._instructions)} instructions")
        return self._instructions
    
    def save(self, path: Union[str, Path]) -> None:
        """Save generated instructions to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [instr.to_dict() for instr in self._instructions]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(self._instructions)} instructions to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> List[InstructionExample]:
        """Load instructions from file."""
        path = Path(path)
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [
            InstructionExample(
                instruction=d['instruction'],
                input=d.get('input', ''),
                output=d.get('output', ''),
                category=d.get('category', ''),
                metadata=d.get('metadata', {}),
            )
            for d in data
        ]


class SyntheticDataPipeline:
    """
    Complete Synthetic Data Pipeline.
    
    Orchestrates the full synthetic data generation workflow:
    1. Seed task generation
    2. Instruction generation
    3. Response generation
    4. Quality filtering
    5. Format conversion
    
    Example:
        >>> pipeline = SyntheticDataPipeline(model, tokenizer)
        >>> dataset = pipeline.run(target_size=10000)
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        output_dir: str = './synthetic_data',
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.self_instruct = SelfInstruct(model, tokenizer)
    
    def run(
        self,
        target_size: int = 10000,
        categories: Optional[List[str]] = None,
        quality_filter: bool = True,
    ) -> List[InstructionExample]:
        """
        Run the full synthetic data pipeline.
        
        Args:
            target_size: Target dataset size
            categories: Optional category filter
            quality_filter: Whether to apply quality filtering
        
        Returns:
            List of InstructionExample
        """
        logger.info(f"Starting synthetic data pipeline (target: {target_size})")
        
        # Generate instructions
        instructions = self.self_instruct.run()
        
        # Filter by categories if specified
        if categories:
            instructions = [
                i for i in instructions
                if i.category in categories
            ]
        
        # Apply quality filtering
        if quality_filter:
            instructions = self._filter_quality(instructions)
        
        # Save results
        self.save(instructions)
        
        return instructions
    
    def _filter_quality(
        self,
        instructions: List[InstructionExample],
    ) -> List[InstructionExample]:
        """Apply quality filtering."""
        filtered = []
        
        for instr in instructions:
            # Basic quality checks
            if len(instr.instruction) < 10:
                continue
            if len(instr.output) < 10:
                continue
            if instr.instruction.lower() in instr.output.lower():
                continue
            
            filtered.append(instr)
        
        logger.info(f"Quality filtering: {len(instructions)} -> {len(filtered)}")
        return filtered
    
    def save(
        self,
        instructions: List[InstructionExample],
        filename: str = 'synthetic_instructions.json',
    ) -> None:
        """Save instructions to file."""
        path = self.output_dir / filename
        
        data = [instr.to_dict() for instr in instructions]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(instructions)} instructions to {path}")
    
    def to_conversations(
        self,
        instructions: List[InstructionExample],
        system_prompt: Optional[str] = None,
    ) -> List[Conversation]:
        """Convert instructions to Conversation format."""
        return [
            instr.to_conversation(system_prompt)
            for instr in instructions
        ]
    
    def to_dataset(
        self,
        instructions: List[InstructionExample],
        template_name: str = 'chatml',
    ) -> 'SyntheticDataset':
        """Convert to PyTorch Dataset."""
        return SyntheticDataset(instructions, self.tokenizer, template_name)


class SyntheticDataset(Dataset):
    """
    PyTorch Dataset for synthetic instructions.
    
    Args:
        instructions: List of InstructionExample
        tokenizer: Tokenizer for encoding
        template_name: Template name for formatting
        max_length: Maximum sequence length
    """
    
    def __init__(
        self,
        instructions: List[InstructionExample],
        tokenizer: Any,
        template_name: str = 'chatml',
        max_length: int = 2048,
    ):
        self.instructions = instructions
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.max_length = max_length
        
        # Pre-process
        self._examples = self._preprocess()
    
    def _preprocess(self) -> List[Dict[str, Any]]:
        """Pre-process instructions for training."""
        examples = []
        
        for instr in self.instructions:
            # Format as conversation
            conv = instr.to_conversation()
            
            # Format with template
            from .formats import ConversationTemplate
            template = ConversationTemplate.get_template(self.template_name)
            text = template.format(conv)
            
            # Tokenize
            encoded = self.tokenizer.encode(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
            )
            
            examples.append({
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': encoded['input_ids'].copy(),
            })
        
        return examples
    
    def __len__(self) -> int:
        return len(self._examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self._examples[idx]
        return {
            'input_ids': torch.tensor(example['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(example['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(example['labels'], dtype=torch.long),
        }
