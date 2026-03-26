"""
Dataset Generator for Arabic LLM Fine-Tuning

This module generates JSONL training datasets from processed book segments
using instruction templates.

Features:
- Generate examples for all roles (tutor, proofreader, poet, muhhaqiq)
- Balance role distribution according to config
- Apply appropriate templates based on content type
- Generate synthetic examples for rare patterns
- Validate and filter examples
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import hashlib

from .schema import (
    TrainingExample, Role, Skill, Level, Domain, Style, TaskType,
    DatasetConfig, DatasetStatistics, validate_example,
    write_jsonl, read_jsonl, compute_statistics
)
from .instruction_templates import (
    get_templates, get_random_template, POETRY_METERS, POETRY_TOPICS,
    ALL_TEMPLATES
)
from .book_processor import TextSegment, BookProcessor


@dataclass
class ExampleGenerator:
    """
    Generate training examples from text segments.
    
    This class applies instruction templates to text segments
    to create diverse training examples for all roles.
    """
    
    config: DatasetConfig
    
    def generate_examples(
        self,
        segments: List[TextSegment],
        target_count: Optional[int] = None,
    ) -> List[TrainingExample]:
        """
        Generate training examples from segments.
        
        Args:
            segments: List of text segments
            target_count: Target number of examples (None = all)
            
        Returns:
            List of TrainingExample objects
        """
        examples = []
        
        # Group segments by type
        by_type = defaultdict(list)
        for seg in segments:
            by_type[seg.segment_type].append(seg)
        
        # Calculate examples per role
        if target_count:
            role_counts = {
                role: int(target_count * ratio)
                for role, ratio in self.config.role_distribution.items()
            }
        else:
            role_counts = {role: len(segments) // 5 for role in self.config.role_distribution}
        
        # Generate examples for each role
        for role, count in role_counts.items():
            role_examples = self._generate_for_role(role, count, by_type)
            examples.extend(role_examples)
        
        # Shuffle examples
        random.shuffle(examples)
        
        return examples
    
    def _generate_for_role(
        self,
        role: str,
        count: int,
        segments_by_type: Dict[str, List[TextSegment]],
    ) -> List[TrainingExample]:
        """Generate examples for a specific role"""
        examples = []
        role_templates = ALL_TEMPLATES.get(role, [])
        
        if not role_templates:
            return examples
        
        # Get segments suitable for this role
        suitable_segments = self._get_suitable_segments(role, segments_by_type)
        
        if not suitable_segments:
            return examples
        
        # Generate examples
        attempts = 0
        max_attempts = count * 3  # Allow for some failures
        
        while len(examples) < count and attempts < max_attempts:
            attempts += 1
            
            # Select random segment and template
            segment = random.choice(suitable_segments)
            template = random.choice(role_templates)
            
            # Generate example
            example = self._create_example(role, template, segment)
            
            if example:
                errors = validate_example(example)
                if not errors:
                    examples.append(example)
        
        return examples
    
    def _get_suitable_segments(
        self,
        role: str,
        segments_by_type: Dict[str, List[TextSegment]],
    ) -> List[TextSegment]:
        """Get segments suitable for a role"""
        suitable = []
        
        if role == "tutor":
            # All segment types work for tutor
            for segs in segments_by_type.values():
                suitable.extend(segs)
        elif role == "proofreader":
            # Prose and hadith for correction
            suitable.extend(segments_by_type.get("prose", []))
            suitable.extend(segments_by_type.get("hadith", []))
        elif role == "poet":
            # Poetry segments
            suitable.extend(segments_by_type.get("poetry", []))
        elif role == "muhhaqiq":
            # Classical prose and hadith
            suitable.extend(segments_by_type.get("prose", []))
            suitable.extend(segments_by_type.get("hadith", []))
            suitable.extend(segments_by_type.get("verse", []))
        elif role == "assistant_general":
            # All types
            for segs in segments_by_type.values():
                suitable.extend(segs)
        
        return suitable
    
    def _create_example(
        self,
        role: str,
        template,
        segment: TextSegment,
    ) -> Optional[TrainingExample]:
        """Create a single training example"""
        try:
            # Determine skill based on template and content
            skill = self._determine_skill(template, segment)
            level = self._determine_level(segment)
            
            # Format instruction
            instruction = self._format_instruction(template, segment)
            
            # Generate output (placeholder - would need LLM or manual annotation)
            output = self._generate_output(template, segment)
            
            # Determine domain and style
            domain_style = self._get_domain_style(segment)
            
            # Create example
            example = TrainingExample(
                instruction=instruction,
                input=segment.text,
                output=output,
                role=Role(role),
                skills=[skill],
                level=level,
                domain=domain_style["domain"],
                style=domain_style["style"],
                task_type=self._get_task_type(template),
                difficulty=self._get_difficulty(level, segment),
                source="extracted_books",
                tags=template.tags + [segment.segment_type],
                book_id=segment.book_id,
                book_title=segment.book_title,
                author_name=segment.author_name,
            )
            
            return example
            
        except Exception as e:
            print(f"Error creating example: {e}")
            return None
    
    def _determine_skill(self, template, segment: TextSegment) -> Skill:
        """Determine skill based on template and segment"""
        # Use template's skill as base
        return Skill(template.skill)
    
    def _determine_level(self, segment: TextSegment) -> Level:
        """Determine difficulty level based on segment"""
        text_len = len(segment.text)
        
        if text_len < 100:
            return Level.BEGINNER
        elif text_len < 300:
            return Level.INTERMEDIATE
        else:
            return Level.ADVANCED
    
    def _format_instruction(self, template, segment: TextSegment) -> str:
        """Format instruction template with segment data"""
        # Prepare variables for template
        kwargs = {
            "sentence": segment.text[:200],
            "text": segment.text[:500],
            "word": segment.text.split()[0] if segment.text.split() else "كلمة",
            "verse": segment.text[:300],
            "quote": segment.text[:150],
            "topic": random.choice(POETRY_TOPICS),
            "meter": random.choice(POETRY_METERS),
            "sentence1": segment.text[:100],
            "sentence2": segment.text[100:200] if len(segment.text) > 100 else segment.text[:100],
            "word1": "كلمة",
            "word2": "كلمة",
            "version1": segment.text[:200],
            "version2": segment.text[:200],
            "partial_verse": segment.text[:50],
            "poem": segment.text[:400],
            "verb": "فعل",
        }
        
        return template.format_instruction(**kwargs)
    
    def _generate_output(self, template, segment: TextSegment) -> str:
        """
        Generate expected output for the example.
        
        Note: In production, this would be:
        1. Manually annotated by experts
        2. Generated by a stronger LLM and reviewed
        3. Extracted from existing educational materials
        
        For now, we generate placeholder outputs.
        """
        role = template.role
        skill = template.skill
        
        if role == "tutor":
            return self._generate_tutor_output(skill, segment)
        elif role == "proofreader":
            return self._generate_proofreader_output(skill, segment)
        elif role == "poet":
            return self._generate_poet_output(skill, segment)
        elif role == "muhhaqiq":
            return self._generate_muhhaqiq_output(skill, segment)
        else:
            return self._generate_assistant_output(skill, segment)
    
    def _generate_tutor_output(self, skill: str, segment: TextSegment) -> str:
        """Generate tutor role output"""
        if skill == "nahw":
            return f"الإعراب: [يتم إعراب الجملة كلمة بكلمة]\n\nالشرح: [يتم شرح القواعد النحوية]"
        elif skill == "balagha":
            return f"نوع الصورة البلاغية: [تشبيه/استعارة/كناية]\n\nالشرح: [يتم شرح الصورة وأثرها]"
        elif skill == "sarf":
            return f"الوزن: [على وزن فعل]\n\nالمادة: [الجذر الثلاثي]"
        else:
            return f"الإجابة: [يتم تقديم الإجابة التعليمية]"
    
    def _generate_proofreader_output(self, skill: str, segment: TextSegment) -> str:
        """Generate proofreader role output"""
        return f"النص المصحَّح: [النص بعد التصحيح]\n\nالأخطاء المصحَّحة:\n1) [نوع الخطأ]: ...\n2) [نوع الخطأ]: ..."
    
    def _generate_poet_output(self, skill: str, segment: TextSegment) -> str:
        """Generate poet role output"""
        if skill == "poetry":
            return "البيت:\n[يتم نظم البيت الشعري الموزون]\n\nالبحر: [اسم البحر]"
        else:
            return "التحليل: [يتم تحليل النص الشعري]"
    
    def _generate_muhhaqiq_output(self, skill: str, segment: TextSegment) -> str:
        """Generate muhhaqiq role output"""
        return f"1) الإعراب: [إعراب الكلمات الرئيسة]\n\n2) المعنى: [شرح المعنى العام]\n\n3) الفوائد: [الفوائد اللغوية]"
    
    def _generate_assistant_output(self, skill: str, segment: TextSegment) -> str:
        """Generate assistant role output"""
        return f"الإجابة: [يتم تقديم الإجابة المفيدة]"
    
    def _get_domain_style(self, segment: TextSegment) -> Dict:
        """Get domain and style based on segment category"""
        category_mapping = {
            "كتب اللغة": {"domain": Domain.LITERATURE, "style": Style.FUSHА_CLASSICAL},
            "التفسير": {"domain": Domain.ISLAMIC_STUDIES, "style": Style.FUSHА_CLASSICAL},
            "كتب السنة": {"domain": Domain.ISLAMIC_STUDIES, "style": Style.FUSHА_CLASSICAL},
            "الأدب": {"domain": Domain.LITERATURE, "style": Style.FUSHА_CLASSICAL},
            "الشعر": {"domain": Domain.LITERATURE, "style": Style.FUSHА_CLASSICAL},
            "البلاغة": {"domain": Domain.EDUCATION, "style": Style.FUSHА_CLASSICAL},
            "النحو": {"domain": Domain.EDUCATION, "style": Style.FUSHА_CLASSICAL},
        }
        
        return category_mapping.get(
            segment.category,
            {"domain": Domain.GENERAL, "style": Style.FUSHА_CLASSICAL}
        )
    
    def _get_task_type(self, template) -> TaskType:
        """Get task type from template"""
        task_mapping = {
            "explanation": TaskType.EXPLANATION,
            "qa": TaskType.QA,
            "correction": TaskType.CORRECTION,
            "rewrite": TaskType.REWRITE,
            "analysis": TaskType.ANALYSIS_AND_EXPLANATION,
            "generation": TaskType.GENERATION,
            "summarization": TaskType.SUMMARIZATION,
        }
        
        # Infer from template tags
        for tag in template.tags:
            if tag in task_mapping:
                return task_mapping[tag]
        
        return TaskType.EXPLANATION
    
    def _get_difficulty(self, level: Level, segment: TextSegment) -> int:
        """Get difficulty score (1-5)"""
        base = {"beginner": 1, "intermediate": 2, "advanced": 3}[level.value]
        
        # Adjust based on text complexity
        if len(segment.text) > 500:
            base += 1
        if segment.segment_type == "verse":
            base += 1
        
        return min(5, max(1, base))


class DatasetGenerator:
    """
    Main dataset generator class.
    
    Orchestrates the full pipeline from books to JSONL dataset.
    """
    
    def __init__(
        self,
        books_dir: str,
        metadata_dir: str,
        output_dir: str,
        config: DatasetConfig,
    ):
        """
        Initialize the dataset generator.
        
        Args:
            books_dir: Path to extracted books
            metadata_dir: Path to metadata
            output_dir: Path to save generated datasets
            config: Dataset configuration
        """
        self.books_dir = Path(books_dir)
        self.metadata_dir = Path(metadata_dir)
        self.output_dir = Path(output_dir)
        self.config = config
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.processor = BookProcessor(books_dir, metadata_dir, str(output_dir / "raw"))
        self.example_generator = ExampleGenerator(config)
    
    def generate(
        self,
        target_examples: int = 50000,
        max_books: int = 1000,
    ) -> DatasetStatistics:
        """
        Generate the complete dataset.
        
        Args:
            target_examples: Target number of training examples
            max_books: Maximum books to process
            
        Returns:
            DatasetStatistics
        """
        print("=" * 60)
        print("Arabic LLM Dataset Generator")
        print("=" * 60)
        
        # Step 1: Load metadata
        print("\n[1/4] Loading book metadata...")
        num_books = self.processor.load_metadata()
        print(f"  Loaded {num_books} books")
        
        # Step 2: Process books
        print("\n[2/4] Processing books...")
        segments = list(self.processor.process_books(
            categories=self.config.source_categories,
            max_books=max_books,
        ))
        print(f"  Extracted {len(segments)} segments")
        
        # Step 3: Generate examples
        print("\n[3/4] Generating training examples...")
        examples = self.example_generator.generate_examples(
            segments,
            target_count=target_examples,
        )
        print(f"  Generated {len(examples)} examples")
        
        # Step 4: Save dataset
        print("\n[4/4] Saving dataset...")
        output_file = self.output_dir / "training_data.jsonl"
        write_jsonl(examples, str(output_file))
        print(f"  Saved to {output_file}")
        
        # Compute statistics
        stats = compute_statistics(examples)
        
        # Save statistics
        stats_file = self.output_dir / "dataset_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Dataset Generation Complete!")
        print("=" * 60)
        print(f"\nTotal Examples: {stats.total_examples}")
        print(f"\nRole Distribution:")
        for role, count in stats.role_counts.items():
            pct = count / stats.total_examples * 100
            print(f"  {role}: {count} ({pct:.1f}%)")
        print(f"\nSkill Distribution:")
        for skill, count in stats.skill_counts.items():
            print(f"  {skill}: {count}")
        print(f"\nLevel Distribution:")
        for level, count in stats.level_counts.items():
            pct = count / stats.total_examples * 100
            print(f"  {level}: {count} ({pct:.1f}%)")
        print(f"\nSource Books: {stats.source_books}")
        print(f"Avg Difficulty: {stats.avg_difficulty:.2f}")
        
        return stats
    
    def generate_split_datasets(
        self,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
    ) -> Dict[str, int]:
        """
        Generate train/val/test split datasets.
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            
        Returns:
            Dict with counts for each split
        """
        # Load full dataset
        full_path = self.output_dir / "training_data.jsonl"
        examples = read_jsonl(str(full_path))
        
        # Shuffle
        random.shuffle(examples)
        
        # Split
        total = len(examples)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        train_examples = examples[:train_end]
        val_examples = examples[train_end:val_end]
        test_examples = examples[val_end:]
        
        # Save splits
        splits = {
            "train": train_examples,
            "val": val_examples,
            "test": test_examples,
        }
        
        counts = {}
        for name, exs in splits.items():
            output_path = self.output_dir / f"{name}.jsonl"
            count = write_jsonl(exs, str(output_path))
            counts[name] = count
            print(f"  {name}: {count} examples")
        
        return counts


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Arabic LLM training dataset")
    parser.add_argument("--books-dir", required=True, help="Path to extracted books")
    parser.add_argument("--metadata-dir", required=True, help="Path to metadata")
    parser.add_argument("--output-dir", default="data/jsonl", help="Output directory")
    parser.add_argument("--target-examples", type=int, default=50000, help="Target examples")
    parser.add_argument("--max-books", type=int, default=1000, help="Max books to process")
    
    args = parser.parse_args()
    
    # Create config
    config = DatasetConfig(
        target_examples=args.target_examples,
    )
    
    # Generate dataset
    generator = DatasetGenerator(
        books_dir=args.books_dir,
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
        config=config,
    )
    
    stats = generator.generate(
        target_examples=args.target_examples,
        max_books=args.max_books,
    )
    
    # Generate splits
    print("\nGenerating train/val/test splits...")
    counts = generator.generate_split_datasets()
    print(f"Split counts: {counts}")


if __name__ == "__main__":
    main()
