"""
Refine Balygh SFT Dataset with LLM

Takes raw SFT examples (with output_stub placeholders) and uses a powerful LLM
(DeepSeek, Qwen2.5-72B, etc.) to generate high-quality actual outputs.

Based on llm_arabic_plan.md implementation plan (lines 8000-9866)

Usage:
    python refine_balygh_sft_with_llm.py --max-examples 1000
"""

import os
import sys
import json
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. Install with: pip install openai")

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    print("Warning: tenacity not installed. Install with: pip install tenacity")

# Paths
RAW_PATH = Path("datasets/jsonl/balygh_sft_from_books.jsonl")
OUT_PATH = Path("datasets/jsonl/balygh_sft_refined.jsonl")
LOG_PATH = Path("logs/refine_errors.log")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# LLM Client Setup
# =============================================================================

def setup_llm_client(provider: str = "deepseek") -> OpenAI:
    """
    Setup LLM client based on provider.
    
    Args:
        provider: "deepseek", "together", "groq", "openai"
    
    Returns:
        OpenAI-compatible client
    """
    if provider == "deepseek":
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        model_name = "deepseek-chat"
    
    elif provider == "together":
        client = OpenAI(
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1"
        )
        model_name = "Qwen/Qwen2.5-72B-Instruct-Turbo"
    
    elif provider == "groq":
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        model_name = "llama-3.1-70b-versatile"
    
    elif provider == "openai":
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        model_name = "gpt-4o-mini"
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    logger.info(f"Using {provider} with model: {model_name}")
    
    return client, model_name


# =============================================================================
# Prompt Building
# =============================================================================

def build_prompt(ex: Dict) -> str:
    """
    Build prompt for LLM to refine SFT example output.
    
    Args:
        ex: Raw SFT example dict
    
    Returns:
        Formatted prompt string
    """
    role = ex.get("role", "")
    skills = ex.get("skills", [])
    instruction = ex.get("instruction", "")
    input_text = ex.get("input", "")
    
    # Add role-specific rules
    if role == "fatwa_assistant_safe":
        extra_rules = (
            "القواعد الصارمة:\n"
            "- التزم قدر الإمكان بما في النص إذا كان موجودًا في input.\n"
            "- إن ذكرت أقوال المذاهب فاذكرها بإيجاز دون ادعاء إجماع إن لم يكن ظاهرًا.\n"
            "- اختم الجواب دائمًا بجملة تنبيه واضحة مثل:\n"
            "  «هذه المعلومات للاستئناس العام، وللفتوى في حالة معيّنة يُرجى مراجعة دار الإفتاء المختصة».\n"
        )
    
    elif role == "muhaddith":
        extra_rules = (
            "القواعد:\n"
            "- اذكر درجة الحديث (صحيح، حسن، ضعيف...) بناءً على ما هو مشهور في كتب الحديث إن أمكن.\n"
            "- لا تخترع أحاديث جديدة ولا تنسب الحديث لكتاب غير مشهور.\n"
            "- إذا لم تكن تعرف درجة الحديث فقل: 'الله أعلم بحاله' أو 'يحتاج إلى تخريج'.\n"
        )
    
    elif role == "mufassir":
        extra_rules = (
            "القواعد:\n"
            "- التزم بالتفسير المأثور (بالنقل) ما أمكن.\n"
            "- لا تُفسّر القرآن بالرأي المجرد.\n"
            "- اذكر أقوال المفسرين المعتمدين (الطبري، القرطبي، ابن كثير، السعدي...).\n"
        )
    
    elif role == "tutor" and "nahw" in skills:
        extra_rules = (
            "القواعد:\n"
            "- أعطِ إعرابًا تفصيليًا كلمةً كلمة.\n"
            "- اذكر نوع كل كلمة وإعرابها وعلامة الإعراب (ظاهرة/مقدرة).\n"
            "- بعد الإعراب، اذكر القاعدة النحوية الأساسية بإيجاز.\n"
        )
    
    elif role == "tutor" and "balagha" in skills:
        extra_rules = (
            "القواعد:\n"
            "- استخرج مثالًا واحدًا واضحًا لصورة بلاغية من النص.\n"
            "- حدّد نوعها (تشبيه/استعارة/كناية) واذكر وجه الشبه أو العلاقة.\n"
            "- بيّن نوع التشبيه (تام/مجمل/مفصل/بليغ) أو الاستعارة (تصريحية/مكنية/تخيلية).\n"
        )
    
    elif role == "faqih":
        extra_rules = (
            "القواعد:\n"
            "- اعرض أقوال المذاهب الأربعة إن وُجدت في النص.\n"
            "- بيّن دليل كل قول إن ذُكر.\n"
            "- إن رجّحت قولًا فبيّن سبب الترجيح.\n"
        )
    
    else:
        extra_rules = (
            "القواعد العامة:\n"
            "- أجب بالعربية الفصحى الواضحة.\n"
            "- إذا كان input نصًا فاعتمد عليه قدر الإمكان ولا تُكثر من الإضافات من خارج النص.\n"
            "- نظّم الإجابة في فقرات واضحة مع عناوين فرعية إن لزم.\n"
        )
    
    prompt = f"""أنت خبير عربي متخصص في هذا الدور: {role}، والمهارات: {", ".join(skills)}.

مهمتك:
- قراءة التعليمات التالية والإجابة عنها بدقة عالية.
- إنتاج مخرَج (output) واحد فقط بصيغة نص عربي منسَّق، بدون JSON أو شروحات خارجية.
- الالتزام التام بالقواعد الخاصة بالدور المذكورة أدناه.

التعليمات (instruction):
{instruction}

النص/المدخل (input):
{input_text}

{extra_rules}
اكتب الآن الجواب (output) النهائي فقط:
"""
    
    return prompt


# =============================================================================
# LLM Call with Retry
# =============================================================================

if TENACITY_AVAILABLE:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def call_llm(client: OpenAI, model_name: str, prompt: str) -> str:
        """
        Call LLM with retry logic.
        
        Args:
            client: OpenAI client
            model_name: Model name
            prompt: Prompt string
        
        Returns:
            Generated text
        """
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.35,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()
else:
    def call_llm(client: OpenAI, model_name: str, prompt: str) -> str:
        """Call LLM without retry"""
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.35,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()


# =============================================================================
# Example Refinement
# =============================================================================

def refine_example(ex: Dict, client: OpenAI, model_name: str) -> Dict:
    """
    Refine a single SFT example using LLM.
    
    Args:
        ex: Raw SFT example
        client: OpenAI client
        model_name: Model name
    
    Returns:
        Refined example dict
    """
    prompt = build_prompt(ex)
    output = call_llm(client, model_name, prompt)
    
    # Clean output (remove any "output:" prefixes)
    output = re.sub(r"^output\s*[:：]\s*", "", output, flags=re.I).strip()
    
    # Remove any markdown code blocks
    output = re.sub(r"^```.*?\n", "", output, flags=re.DOTALL)
    output = re.sub(r"\n```$", "", output)
    
    ex["output"] = output
    ex["quality_score"] = 0.9  # Placeholder; can be improved with actual scoring
    ex["refined_at"] = datetime.now().isoformat()
    ex["refined_by"] = model_name
    
    return ex


# =============================================================================
# Main Pipeline
# =============================================================================

def main(
    max_examples: Optional[int] = None,
    resume: bool = True,
    provider: str = "deepseek",
):
    """
    Main refinement pipeline.
    
    Args:
        max_examples: Maximum examples to refine (None = all)
        resume: Whether to resume from existing output file
        provider: LLM provider to use
    """
    if not OPENAI_AVAILABLE:
        logger.error("openai library not installed. Install with: pip install openai")
        return
    
    # Setup client
    client, model_name = setup_llm_client(provider)
    
    # Load already processed IDs if resuming
    processed_ids = set()
    if resume and OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "id" in obj:
                    processed_ids.add(obj["id"])
        
        logger.info(f"Resuming: {len(processed_ids)} examples already processed")
    
    # Open files
    if not RAW_PATH.exists():
        logger.error(f"Raw file not found: {RAW_PATH}")
        logger.error("Run build_balygh_sft_dataset.py first")
        return
    
    in_f = open(RAW_PATH, encoding="utf-8")
    out_f = open(OUT_PATH, "a", encoding="utf-8")
    
    count = 0
    error_count = 0
    
    logger.info(f"Starting refinement with {model_name}...")
    
    for line in in_f:
        if not line.strip():
            continue
        
        ex = json.loads(line)
        
        # Assign ID if not present
        if "id" not in ex:
            ex["id"] = f"{ex.get('role','ex')}_{count:06d}"
        
        # Skip if already processed
        if ex["id"] in processed_ids:
            continue
        
        try:
            refined = refine_example(ex, client, model_name)
            out_f.write(json.dumps(refined, ensure_ascii=False) + "\n")
            out_f.flush()
            
            count += 1
            processed_ids.add(ex["id"])
            
            if count % 10 == 0:
                logger.info(f"Refined {count} examples...")
            
            if max_examples and count >= max_examples:
                break
        
        except Exception as e:
            error_count += 1
            logger.error(f"Error refining {ex.get('id')}: {str(e)}")
            
            # Log error for later review
            with open(LOG_PATH, "a", encoding="utf-8") as log_f:
                log_f.write(json.dumps({"id": ex.get("id"), "error": str(e)}, ensure_ascii=False) + "\n")
            
            continue
    
    in_f.close()
    out_f.close()
    
    logger.info("=" * 60)
    logger.info(f"✅ Refinement Complete!")
    logger.info(f"   Refined: {count} examples")
    logger.info(f"   Errors: {error_count}")
    logger.info(f"   Output: {OUT_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine Balygh SFT Dataset with LLM")
    parser.add_argument("--max-examples", type=int, help="Max examples to refine (default: all)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "together", "groq", "openai"])
    parser.add_argument("--no-resume", action="store_true", help="Don't resume, start fresh")
    
    args = parser.parse_args()
    
    main(
        max_examples=args.max_examples,
        resume=not args.no_resume,
        provider=args.provider,
    )
