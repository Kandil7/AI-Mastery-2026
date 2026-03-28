"""
Test all arabic_llm modules

Run this to verify all modules are working correctly.
"""

import warnings
warnings.filterwarnings('ignore')

def test_module(name, import_stmt):
    """Test a module import"""
    try:
        exec(import_stmt)
        print(f"✅ {name}")
        return True
    except Exception as e:
        print(f"❌ {name}: {str(e)[:100]}")
        return False

print("=" * 70)
print("Testing arabic_llm Modules")
print("=" * 70)
print()

results = []

# Core modules
print("Core Modules:")
results.append(test_module("core.schema", "from arabic_llm.core.schema import Role, Skill"))
results.append(test_module("core.templates", "from arabic_llm.core.templates import Template"))
results.append(test_module("core.book_processor", "from arabic_llm.core.book_processor import BookProcessor"))
results.append(test_module("core.dataset_generator", "from arabic_llm.core.dataset_generator import ExampleGenerator"))
print()

# Processing modules (in pipeline/)
print("Processing Modules:")
results.append(test_module("pipeline.cleaning", "from arabic_llm.pipeline.cleaning import ArabicTextCleaner"))
results.append(test_module("pipeline.deduplication", "from arabic_llm.pipeline.deduplication import ExactDeduplicator"))
print()

# Generation modules (in core/)
print("Generation Modules:")
results.append(test_module("core.dataset_generator (generation)", "from arabic_llm.core.dataset_generator import DatasetGenerator"))
print()

# Training modules (in models/)
print("Training Modules:")
results.append(test_module("models.qlora", "from arabic_llm.models.qlora import QLoRAConfig"))
results.append(test_module("models.quantization", "from arabic_llm.models.quantization import QuantizationConfig"))
print()

# Agents (optional - requires torch)
print("Agents (optional - requires torch):")
results.append(test_module("agents.data_collector", "from arabic_llm.agents.data_collector import DataCollectionAgent"))
print()

# Integration modules
print("Integration Modules:")
results.append(test_module("integration.databases", "from arabic_llm.integration.databases import DatabaseConnection"))
results.append(test_module("integration.system_books", "from arabic_llm.integration.system_books import SystemBookIntegration"))
print()

# Utils modules
print("Utils Modules:")
results.append(test_module("utils.arabic", "from arabic_llm.utils.arabic import get_arabic_ratio"))
results.append(test_module("utils.io", "from arabic_llm.utils.io import read_jsonl"))
results.append(test_module("utils.logging", "from arabic_llm.utils.logging import setup_logging"))
results.append(test_module("utils.text", "from arabic_llm.utils.text import count_words"))
print()

# Summary
print("=" * 70)
print(f"Results: {sum(results)}/{len(results)} modules working")
print("=" * 70)

if all(results):
    print("\n🎉 ALL MODULES WORKING!")
    exit(0)
else:
    print(f"\n⚠️  {len(results) - sum(results)} modules have issues")
    exit(1)
