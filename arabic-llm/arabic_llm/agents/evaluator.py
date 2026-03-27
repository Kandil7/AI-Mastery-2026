"""
Balygh (بليغ) Evaluation Suite

Comprehensive evaluation for Arabic LLM with 29 roles and 76 skills.

Includes:
1. OALL Benchmarks (Open Arabic LLM Leaderboard)
2. Custom Arabic Linguistics Tests
3. Role-Specific Evaluations
4. Islamic Sciences Assessments
5. RAG Groundedness Evaluation

Based on llm_arabic_plan.md evaluation section
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import re

# ML libraries
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    import warnings
    warnings.warn("transformers not installed. Install with: pip install transformers peft")

# Metrics
try:
    from sacrebleu import corpus_bleu
    from rouge_score import rouge_scorer
    import evaluate
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class EvaluationExample:
    """A single evaluation example"""
    id: str
    instruction: str
    input: str
    expected_output: str
    role: str
    skills: List[str]
    level: str
    domain: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating a single example"""
    example_id: str
    role: str
    predicted_output: str
    expected_output: str
    metrics: Dict[str, float] = field(default_factory=dict)
    human_rating: Optional[float] = None  # 1-5 scale
    notes: str = ""


@dataclass
class BenchmarkResults:
    """Results for a complete benchmark"""
    benchmark_name: str
    total_examples: int
    evaluated_examples: int
    metrics: Dict[str, float] = field(default_factory=dict)
    role_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    skill_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: str = ""
    
    def to_dict(self) -> dict:
        return {
            "benchmark_name": self.benchmark_name,
            "total_examples": self.total_examples,
            "evaluated_examples": self.evaluated_examples,
            "metrics": self.metrics,
            "role_breakdown": self.role_breakdown,
            "skill_breakdown": self.skill_breakdown,
            "timestamp": self.timestamp,
        }


# ============================================================================
# METRICS CALCULATOR
# ============================================================================

class MetricsCalculator:
    """
    Calculate evaluation metrics for Arabic text.
    
    Supports:
    - BLEU (via sacrebleu)
    - ROUGE (via rouge_score)
    - BERTScore (via evaluate)
    - Exact Match
    - F1 Score (token-level)
    """
    
    def __init__(self):
        if not METRICS_AVAILABLE:
            raise ImportError(
                "Metrics require sacrebleu, rouge_score, and evaluate. "
                "Install with: pip install sacrebleu rouge_score evaluate"
            )
        
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    def calculate_bleu(self, predictions: List[str], references: List[List[str]]) -> float:
        """Calculate BLEU score"""
        try:
            bleu = corpus_bleu(predictions, [references])
            return bleu.score
        except Exception as e:
            logger.error(f"BLEU calculation error: {e}")
            return 0.0
    
    def calculate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, prediction)
            return {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        except Exception as e:
            logger.error(f"ROUGE calculation error: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def calculate_exact_match(self, prediction: str, reference: str) -> float:
        """Calculate exact match (binary)"""
        return 1.0 if prediction.strip() == reference.strip() else 0.0
    
    def calculate_f1(self, prediction: str, reference: str) -> float:
        """Calculate token-level F1 score"""
        pred_tokens = set(prediction.split())
        ref_tokens = set(reference.split())
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        intersection = pred_tokens & ref_tokens
        
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def calculate_all_metrics(
        self, 
        prediction: str, 
        reference: str,
        calculate_bleu: bool = False
    ) -> Dict[str, float]:
        """Calculate all metrics for a single example"""
        metrics = {}
        
        # Exact match
        metrics["exact_match"] = self.calculate_exact_match(prediction, reference)
        
        # F1 score
        metrics["f1"] = self.calculate_f1(prediction, reference)
        
        # ROUGE
        rouge_scores = self.calculate_rouge(prediction, reference)
        metrics.update(rouge_scores)
        
        return metrics


# ============================================================================
# BENCHMARK DATASETS
# ============================================================================

class BenchmarkDatasets:
    """
    Load and manage benchmark datasets.
    
    Includes:
    1. OALL Benchmarks
    2. Custom Arabic Linguistics Tests
    3. Role-Specific Evaluations
    """
    
    def __init__(self, data_dir: str = "datasets/evaluation"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_oall_benchmarks(self) -> List[EvaluationExample]:
        """
        Load Open Arabic LLM Leaderboard (OALL) benchmarks.
        
        Includes:
        - Arabic MMLU
        - Arabic QA
        - Arabic Summarization
        - Arabic Translation
        """
        examples = []
        
        # Arabic MMLU (Multiple Choice)
        mmlu_file = self.data_dir / "arabic_mmlu.jsonl"
        if mmlu_file.exists():
            with open(mmlu_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    examples.append(EvaluationExample(
                        id=f"oall-mmlu-{i}",
                        instruction=data.get("question", ""),
                        input="",
                        expected_output=data.get("answer", ""),
                        role="assistant_general",
                        skills=["qa"],
                        level="advanced",
                        domain="academic",
                        metadata={"type": "multiple_choice"}
                    ))
        
        # Arabic QA
        qa_file = self.data_dir / "arabic_qa.jsonl"
        if qa_file.exists():
            with open(qa_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    examples.append(EvaluationExample(
                        id=f"oall-qa-{i}",
                        instruction=data.get("question", ""),
                        input=data.get("context", ""),
                        expected_output=data.get("answer", ""),
                        role="rag_assistant",
                        skills=["rag_grounded_answering"],
                        level="intermediate",
                        domain="general",
                        metadata={"type": "extractive_qa"}
                    ))
        
        # Arabic Summarization
        summary_file = self.data_dir / "arabic_summarization.jsonl"
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    examples.append(EvaluationExample(
                        id=f"oall-summary-{i}",
                        instruction="لخّص النص التالي:",
                        input=data.get("text", ""),
                        expected_output=data.get("summary", ""),
                        role="summarizer_ar",
                        skills=["summarization"],
                        level="intermediate",
                        domain="media",
                        metadata={"type": "summarization"}
                    ))
        
        logger.info(f"Loaded {len(examples)} OALL benchmark examples")
        
        return examples
    
    def load_linguistics_tests(self) -> List[EvaluationExample]:
        """
        Load custom Arabic linguistics tests.
        
        Includes:
        - I'rab (إعراب) - Grammar analysis
        - Balagha (بلاغة) - Rhetoric identification
        - Poetry Criticism
        - Error Correction
        """
        examples = []
        
        # I'rab Tests
        i3rab_file = self.data_dir / "i3rab_tests.jsonl"
        if i3rab_file.exists():
            with open(i3rab_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    examples.append(EvaluationExample(
                        id=f"ling-i3rab-{i}",
                        instruction=f"أعرب الجملة التالية إعراباً مفصلاً: {data.get('sentence', '')}",
                        input="",
                        expected_output=data.get("parsing", ""),
                        role="tutor",
                        skills=["nahw"],
                        level="advanced",
                        domain="linguistics",
                        metadata={"type": "grammar_parsing"}
                    ))
        
        # Balagha Tests
        balagha_file = self.data_dir / "balagha_tests.jsonl"
        if balagha_file.exists():
            with open(balogha_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    examples.append(EvaluationExample(
                        id=f"ling-balagha-{i}",
                        instruction=f"بيّن الصور البلاغية في: {data.get('text', '')}",
                        input="",
                        expected_output=data.get("analysis", ""),
                        role="tutor",
                        skills=["balagha"],
                        level="advanced",
                        domain="linguistics",
                        metadata={"type": "rhetoric_analysis"}
                    ))
        
        # Poetry Criticism
        poetry_file = self.data_dir / "poetry_criticism.jsonl"
        if poetry_file.exists():
            with open(poetry_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    examples.append(EvaluationExample(
                        id=f"ling-poetry-{i}",
                        instruction=f"انقد البيت الشعري التالي: {data.get('verse', '')}",
                        input="",
                        expected_output=data.get("criticism", ""),
                        role="poet",
                        skills=["poetry", "literary_criticism"],
                        level="specialist",
                        domain="literature",
                        metadata={"type": "poetry_criticism"}
                    ))
        
        # Error Correction
        error_file = self.data_dir / "error_correction.jsonl"
        if error_file.exists():
            with open(error_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    examples.append(EvaluationExample(
                        id=f"ling-error-{i}",
                        instruction=f"صحّح الأخطاء في النص التالي: {data.get('text', '')}",
                        input="",
                        expected_output=data.get("corrected", ""),
                        role="proofreader",
                        skills=["orthography", "nahw", "error_analysis_ar"],
                        level="intermediate",
                        domain="linguistics",
                        metadata={"type": "error_correction"}
                    ))
        
        logger.info(f"Loaded {len(examples)} linguistics test examples")
        
        return examples
    
    def load_islamic_sciences_tests(self) -> List[EvaluationExample]:
        """
        Load Islamic sciences evaluations.
        
        Includes:
        - Fiqh (فقه) - Islamic jurisprudence
        - Hadith (حديث) - Prophetic traditions
        - Tafsir (تفسير) - Quranic exegesis
        - Aqeedah (عقيدة) - Islamic creed
        """
        examples = []
        
        # Fiqh Tests
        fiqh_file = self.data_dir / "fiqh_tests.jsonl"
        if fiqh_file.exists():
            with open(fiqh_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    examples.append(EvaluationExample(
                        id=f"islamic-fiqh-{i}",
                        instruction=f"ما حكم {data.get('topic', '')} في الفقه الإسلامي؟",
                        input="",
                        expected_output=data.get("ruling", ""),
                        role="faqih",
                        skills=["fiqh", "usul_fiqh"],
                        level="advanced",
                        domain="islamic_studies",
                        metadata={"type": "fiqh_ruling"}
                    ))
        
        # Hadith Tests
        hadith_file = self.data_dir / "hadith_tests.jsonl"
        if hadith_file.exists():
            with open(hadith_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    examples.append(EvaluationExample(
                        id=f"islamic-hadith-{i}",
                        instruction=f"خرّج الحديث التالي: {data.get('hadith', '')}",
                        input="",
                        expected_output=data.get("takhreej", ""),
                        role="muhaddith",
                        skills=["hadith", "hadith_mustalah"],
                        level="specialist",
                        domain="islamic_studies",
                        metadata={"type": "hadith_takhreej"}
                    ))
        
        # Tafsir Tests
        tafsir_file = self.data_dir / "tafsir_tests.jsonl"
        if tafsir_file.exists():
            with open(tafsir_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    examples.append(EvaluationExample(
                        id=f"islamic-tafsir-{i}",
                        instruction=f"فسّر الآية الكريمة: {data.get('ayah', '')}",
                        input="",
                        expected_output=data.get("tafsir", ""),
                        role="mufassir",
                        skills=["tafsir", "quran_sciences"],
                        level="advanced",
                        domain="islamic_studies",
                        metadata={"type": "quran_tafsir"}
                    ))
        
        logger.info(f"Loaded {len(examples)} Islamic sciences examples")
        
        return examples
    
    def load_role_specific_tests(self) -> Dict[str, List[EvaluationExample]]:
        """
        Load role-specific evaluation tests for all 29 roles.
        
        Returns:
            Dictionary mapping role names to evaluation examples
        """
        role_examples = {}
        
        roles_dir = self.data_dir / "role_specific"
        if not roles_dir.exists():
            logger.warning(f"Role-specific tests directory not found: {roles_dir}")
            return role_examples
        
        # Load tests for each role
        for role_file in roles_dir.glob("*.jsonl"):
            role_name = role_file.stem
            
            examples = []
            with open(role_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    examples.append(EvaluationExample(
                        id=f"{role_name}-{i}",
                        instruction=data.get("instruction", ""),
                        input=data.get("input", ""),
                        expected_output=data.get("output", ""),
                        role=role_name,
                        skills=data.get("skills", []),
                        level=data.get("level", "intermediate"),
                        domain=data.get("domain", "general"),
                        metadata=data
                    ))
            
            role_examples[role_name] = examples
            logger.info(f"Loaded {len(examples)} examples for role: {role_name}")
        
        return role_examples


# ============================================================================
# MODEL EVALUATOR
# ============================================================================

class ModelEvaluator:
    """
    Evaluate Balygh model on benchmarks.
    
    Usage:
        evaluator = ModelEvaluator(model_path, device="cuda")
        results = evaluator.evaluate_benchmark(benchmark_name, examples)
    """
    
    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        device: str = "cuda",
        max_length: int = 4096,
    ):
        """
        Initialize model evaluator.
        
        Args:
            model_path: Path to base model
            adapter_path: Path to LoRA adapter (optional)
            device: Device to run evaluation on
            max_length: Maximum sequence length
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Evaluation requires transformers and peft. "
                "Install with: pip install transformers peft"
            )
        
        self.device = device
        self.max_length = max_length
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model
        logger.info(f"Loading base model from {model_path}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load adapter if provided
        if adapter_path:
            logger.info(f"Loading LoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        else:
            self.model = self.base_model
        
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            
        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - max_new_tokens,
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def evaluate_example(
        self,
        example: EvaluationExample,
        metrics_calculator: MetricsCalculator,
    ) -> EvaluationResult:
        """
        Evaluate a single example.
        
        Args:
            example: Evaluation example
            metrics_calculator: Metrics calculator instance
            
        Returns:
            Evaluation result
        """
        # Generate prediction
        prompt = f"{example.instruction}\n\n{example.input}" if example.input else example.instruction
        
        predicted_output = self.generate(prompt)
        
        # Calculate metrics
        metrics = metrics_calculator.calculate_all_metrics(
            predicted_output,
            example.expected_output,
        )
        
        # Create result
        result = EvaluationResult(
            example_id=example.id,
            role=example.role,
            predicted_output=predicted_output,
            expected_output=example.expected_output,
            metrics=metrics,
        )
        
        return result
    
    def evaluate_benchmark(
        self,
        benchmark_name: str,
        examples: List[EvaluationExample],
        batch_size: int = 1,
        max_examples: Optional[int] = None,
    ) -> BenchmarkResults:
        """
        Evaluate model on a benchmark.
        
        Args:
            benchmark_name: Name of benchmark
            examples: List of evaluation examples
            batch_size: Batch size (not used yet)
            max_examples: Maximum examples to evaluate
            
        Returns:
            Benchmark results
        """
        logger.info(f"Evaluating {benchmark_name} with {len(examples)} examples...")
        
        metrics_calculator = MetricsCalculator()
        
        results: List[EvaluationResult] = []
        role_metrics: Dict[str, List[Dict]] = {}
        skill_metrics: Dict[str, List[Dict]] = {}
        
        # Limit examples if specified
        if max_examples:
            examples = examples[:max_examples]
        
        # Evaluate each example
        for i, example in enumerate(examples):
            if (i + 1) % 10 == 0:
                logger.info(f"Evaluating example {i+1}/{len(examples)}...")
            
            result = self.evaluate_example(example, metrics_calculator)
            results.append(result)
            
            # Aggregate by role
            if example.role not in role_metrics:
                role_metrics[example.role] = []
            role_metrics[example.role].append(result.metrics)
            
            # Aggregate by skill
            for skill in example.skills:
                if skill not in skill_metrics:
                    skill_metrics[skill] = []
                skill_metrics[skill].append(result.metrics)
        
        # Calculate aggregate metrics
        all_metrics = [r.metrics for r in results]
        
        aggregate_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]
                aggregate_metrics[key] = sum(values) / len(values) if values else 0.0
        
        # Calculate role breakdown
        role_breakdown = {}
        for role, metrics_list in role_metrics.items():
            role_breakdown[role] = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list if key in m]
                role_breakdown[role][key] = sum(values) / len(values) if values else 0.0
        
        # Calculate skill breakdown
        skill_breakdown = {}
        for skill, metrics_list in skill_metrics.items():
            skill_breakdown[skill] = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list if key in m]
                skill_breakdown[skill][key] = sum(values) / len(values) if values else 0.0
        
        # Create benchmark results
        benchmark_results = BenchmarkResults(
            benchmark_name=benchmark_name,
            total_examples=len(examples),
            evaluated_examples=len(results),
            metrics=aggregate_metrics,
            role_breakdown=role_breakdown,
            skill_breakdown=skill_breakdown,
            timestamp=datetime.now().isoformat(),
        )
        
        logger.info(f"Benchmark {benchmark_name} complete. Average F1: {aggregate_metrics.get('f1', 0):.3f}")
        
        return benchmark_results


# ============================================================================
# EVALUATION RUNNER
# ============================================================================

class EvaluationRunner:
    """
    Main evaluation runner for Balygh.
    
    Coordinates all benchmarks and produces comprehensive report.
    
    Usage:
        runner = EvaluationRunner(model_path, adapter_path)
        results = runner.run_full_evaluation(output_dir="evaluation/results")
    """
    
    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize evaluation runner.
        
        Args:
            model_path: Path to base model
            adapter_path: Path to LoRA adapter
            device: Device to run on
        """
        self.evaluator = ModelEvaluator(model_path, adapter_path, device)
        self.datasets = BenchmarkDatasets()
    
    def run_full_evaluation(
        self,
        output_dir: str = "evaluation/results",
        max_examples_per_benchmark: Optional[int] = None,
    ) -> Dict[str, BenchmarkResults]:
        """
        Run full evaluation suite.
        
        Args:
            output_dir: Directory to save results
            max_examples_per_benchmark: Limit examples per benchmark
            
        Returns:
            Dictionary of benchmark results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # 1. OALL Benchmarks
        logger.info("=" * 60)
        logger.info("Running OALL Benchmarks...")
        logger.info("=" * 60)
        
        oall_examples = self.datasets.load_oall_benchmarks()
        if oall_examples:
            oall_results = self.evaluator.evaluate_benchmark(
                "OALL",
                oall_examples,
                max_examples=max_examples_per_benchmark,
            )
            results["OALL"] = oall_results
            self._save_results(oall_results, output_path / "oall_results.json")
        
        # 2. Linguistics Tests
        logger.info("=" * 60)
        logger.info("Running Linguistics Tests...")
        logger.info("=" * 60)
        
        ling_examples = self.datasets.load_linguistics_tests()
        if ling_examples:
            ling_results = self.evaluator.evaluate_benchmark(
                "Arabic_Linguistics",
                ling_examples,
                max_examples=max_examples_per_benchmark,
            )
            results["Arabic_Linguistics"] = ling_results
            self._save_results(ling_results, output_path / "linguistics_results.json")
        
        # 3. Islamic Sciences
        logger.info("=" * 60)
        logger.info("Running Islamic Sciences Evaluation...")
        logger.info("=" * 60)
        
        islamic_examples = self.datasets.load_islamic_sciences_tests()
        if islamic_examples:
            islamic_results = self.evaluator.evaluate_benchmark(
                "Islamic_Sciences",
                islamic_examples,
                max_examples=max_examples_per_benchmark,
            )
            results["Islamic_Sciences"] = islamic_results
            self._save_results(islamic_results, output_path / "islamic_sciences_results.json")
        
        # 4. Role-Specific Tests
        logger.info("=" * 60)
        logger.info("Running Role-Specific Evaluations...")
        logger.info("=" * 60)
        
        role_examples_dict = self.datasets.load_role_specific_tests()
        for role_name, role_examples in role_examples_dict.items():
            logger.info(f"Evaluating role: {role_name} ({len(role_examples)} examples)")
            
            role_results = self.evaluator.evaluate_benchmark(
                f"Role_{role_name}",
                role_examples,
                max_examples=max_examples_per_benchmark,
            )
            results[f"Role_{role_name}"] = role_results
            self._save_results(role_results, output_path / f"role_{role_name}_results.json")
        
        # 5. Summary Report
        self._generate_summary_report(results, output_path / "summary_report.md")
        
        logger.info("=" * 60)
        logger.info(f"Evaluation complete! Results saved to {output_path}")
        logger.info("=" * 60)
        
        return results
    
    def _save_results(self, results: BenchmarkResults, output_file: str):
        """Save benchmark results to JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved results to {output_file}")
    
    def _generate_summary_report(
        self,
        results: Dict[str, BenchmarkResults],
        output_file: str,
    ):
        """Generate markdown summary report"""
        report = "# Balygh (بليغ) Evaluation Summary Report\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## Overall Performance\n\n"
        
        # Aggregate all metrics
        all_f1 = []
        all_rouge1 = []
        
        for benchmark_name, benchmark_results in results.items():
            f1 = benchmark_results.metrics.get("f1", 0)
            rouge1 = benchmark_results.metrics.get("rouge1", 0)
            
            if f1 > 0:
                all_f1.append(f1)
            if rouge1 > 0:
                all_rouge1.append(rouge1)
            
            report += f"### {benchmark_name}\n"
            report += f"- Examples: {benchmark_results.evaluated_examples}/{benchmark_results.total_examples}\n"
            report += f"- F1 Score: {f1:.3f}\n"
            report += f"- ROUGE-1: {rouge1:.3f}\n\n"
        
        # Overall averages
        if all_f1:
            report += f"\n## Average Metrics\n"
            report += f"- **Average F1:** {sum(all_f1) / len(all_f1):.3f}\n"
        if all_rouge1:
            report += f"- **Average ROUGE-1:** {sum(all_rouge1) / len(all_rouge1):.3f}\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Generated summary report: {output_file}")


# ============================================================================
# MAIN - CLI USAGE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Balygh Evaluation Suite")
    parser.add_argument("--model-path", required=True, help="Path to base model")
    parser.add_argument("--adapter-path", help="Path to LoRA adapter")
    parser.add_argument("--output-dir", default="evaluation/results", help="Output directory")
    parser.add_argument("--max-examples", type=int, help="Max examples per benchmark")
    parser.add_argument("--device", default="cuda", help="Device to run on")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run evaluation
    runner = EvaluationRunner(
        args.model_path,
        args.adapter_path,
        args.device,
    )
    
    results = runner.run_full_evaluation(
        args.output_dir,
        args.max_examples,
    )
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    
    for benchmark_name, benchmark_results in results.items():
        print(f"\n{benchmark_name}:")
        print(f"  F1: {benchmark_results.metrics.get('f1', 0):.3f}")
        print(f"  ROUGE-1: {benchmark_results.metrics.get('rouge1', 0):.3f}")
