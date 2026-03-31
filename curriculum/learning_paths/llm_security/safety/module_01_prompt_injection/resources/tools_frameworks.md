# Tools & Frameworks: Prompt Injection Security

**Module:** SEC-SAFETY-001  
**Resource Type:** Tools and Frameworks  
**Last Updated:** March 30, 2026

---

## Table of Contents

1. [Detection Tools](#detection-tools)
2. [Guardrails Frameworks](#guardrails-frameworks)
3. [Testing & Red Team Tools](#testing--red-team-tools)
4. [Monitoring & Logging](#monitoring--logging)
5. [Development Libraries](#development-libraries)
6. [Commercial Solutions](#commercial-solutions)
7. [Tool Comparison Matrix](#tool-comparison-matrix)

---

## Detection Tools

### 1. Lakera Guard
**Type:** API/SDK  
**License:** Commercial (Free tier available)  
**Link:** https://www.lakera.ai/products/lakera-guard

**Features:**
- Real-time prompt injection detection
- PII detection and redaction
- Toxicity and hate speech detection
- Multi-language support

**Integration:**
```python
import lakera

client = lakera.Client(api_key="your-key")

response = client.guard.scan(
    input=user_input,
    categories=["prompt_injection", "pii", "toxicity"]
)

if response.flagged:
    # Handle flagged content
    pass
```

**Best For:** Production applications needing comprehensive content moderation

---

### 2. Rebuff AI
**Type:** Open Source  
**License:** MIT  
**Link:** https://github.com/protectai/rebuff

**Features:**
- Prompt injection detection
- Vector-based anomaly detection
- Self-checking mechanism
- Easy integration with LangChain

**Integration:**
```python
from rebuff import Rebuff

rb = Rebuff(api_key="your-key")

result = rb.detect_injection(prompt=user_input)
if result.is_injection:
    # Handle injection attempt
    pass
```

**Best For:** Open source projects, LangChain applications

---

### 3. Garak (LLM Vulnerability Scanner)
**Type:** Open Source  
**License:** Apache 2.0  
**Link:** https://github.com/leondz/garak

**Features:**
- Automated vulnerability scanning
- 100+ probe types
- Prompt injection testing
- Hallucination detection
- Data leakage testing

**Usage:**
```bash
# Install
pip install garak

# Run scan
garak --model_type openai --model_name gpt-3.5-turbo

# Specific probes
garak --probes prompt_injection.InjectWithSeparator
```

**Best For:** Security testing, red team exercises

---

### 4. PromptInject
**Type:** Open Source  
**License:** MIT  
**Link:** https://github.com/agencyenterprise/PromptInject

**Features:**
- Modular injection testing framework
- Pre-built attack payloads
- Custom probe creation
- Scoring and reporting

**Usage:**
```python
from promptinject import attack_builder

attack = attack_builder.build_attack(
    attack_strategy="ignore_instructions",
    target_model="gpt-3.5-turbo"
)

result = attack.execute()
```

**Best For:** Security research, penetration testing

---

## Guardrails Frameworks

### 1. NVIDIA NeMo Guardrails
**Type:** Open Source  
**License:** Apache 2.0  
**Link:** https://github.com/NVIDIA/NeMo-Guardrails

**Features:**
- Colang language for defining guardrails
- Topic control and flow management
- Input/output filtering
- Hallucination detection
- Fact checking integration

**Integration:**
```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("config_path")
rails = LLMRails(config)

response = rails.generate(
    messages=[{"role": "user", "content": user_input}]
)
```

**Colang Example:**
```colang
define user express greeting
  "hello"
  "hi"
  "hey"

define flow
  user express greeting
  bot express greeting
```

**Best For:** Complex applications needing fine-grained control

---

### 2. Guardrails AI
**Type:** Open Source  
**License:** Apache 2.0  
**Link:** https://github.com/guardrails-ai/guardrails

**Features:**
- Output validation with Pydantic-like schemas
- Built-in validators for common cases
- Retry and correction mechanisms
- LLM-agnostic

**Integration:**
```python
import guardrails as gd

# Define schema with validators
schema = gd.ValidatedPrompt(
    validators=[
        gd.validator.ProfanityFilter(),
        gd.validator.ToxicLanguage(),
        gd.validator.EndsWith(period_count=1)
    ]
)

# Use with LLM
guarded_llm = gd.Guard(schema, llm_provider="openai")
response = guarded_llm.generate(prompt=user_input)
```

**Best For:** Structured output validation, compliance requirements

---

### 3. Microsoft Guidance
**Type:** Open Source  
**License:** MIT  
**Link:** https://github.com/microsoft/guidance

**Features:**
- Control LLM generation with constraints
- Interleaved generation and computation
- Grammar-based constraints
- Prompt injection resistance

**Integration:**
```python
import guidance

@guidance
def secure_response(lm, user_input):
    lm += f"User: {user_input}\n"
    lm += "Assistant: " + guidance.gen(
        "response",
        max_tokens=200,
        stop=["User:", "<injection>"]
    )
    return lm

model = guidance.models.OpenAI("gpt-3.5-turbo")
result = model(secure_response(user_input="Hello"))
```

**Best For:** Controlled generation, structured outputs

---

### 4. LangChain Security Features
**Type:** Open Source  
**License:** MIT  
**Link:** https://python.langchain.com/docs/guides/safety

**Features:**
- Input/output chains for validation
- Integration with moderation APIs
- Custom safety chains
- RAG security patterns

**Integration:**
```python
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser

# Add safety layer
safety_chain = create_safety_chain(
    input_validator=injection_detector,
    output_validator=pii_filter
)

chain = safety_chain | llm_chain
```

**Best For:** LangChain-based applications

---

## Testing & Red Team Tools

### 1. PyRIT (Python Risk Identification Tool)
**Type:** Open Source  
**License:** MIT  
**Link:** https://github.com/Azure/PyRIT

**Features:**
- Automated red teaming
- Multiple attack strategies
- Azure AI integration
- Comprehensive reporting

**Usage:**
```python
from pyrit import RedTeamOrchestrator

orchestrator = RedTeamOrchestrator(
    target_model="gpt-4",
    attack_strategies=["prompt_injection", "jailbreak"]
)

results = orchestrator.run()
results.generate_report()
```

**Best For:** Enterprise red team exercises, Azure AI users

---

### 2. Cyscale LLM Security
**Type:** Commercial  
**License:** Commercial  
**Link:** https://www.cyscale.com/llm-security

**Features:**
- Automated vulnerability assessment
- Compliance checking
- Continuous monitoring
- Enterprise reporting

**Best For:** Enterprise security teams, compliance requirements

---

### 3. LLM Fuzzer
**Type:** Open Source  
**License:** MIT  
**Link:** https://github.com/grayhatwll/llm-fuzzer

**Features:**
- Fuzzing for LLM inputs
- Edge case discovery
- Automated testing
- CI/CD integration

**Usage:**
```bash
llm-fuzzer --target api_endpoint --iterations 1000
```

**Best For:** Automated testing, CI/CD pipelines

---

## Monitoring & Logging

### 1. LangSmith (LangChain)
**Type:** Commercial (Free tier)  
**Link:** https://smith.langchain.com/

**Features:**
- Trace and debug LLM calls
- Monitor for anomalies
- Evaluate outputs
- Track injection attempts

**Integration:**
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"

# All LangChain calls are automatically traced
```

**Best For:** LangChain applications, development and production monitoring

---

### 2. Arize Phoenix
**Type:** Open Source  
**License:** Apache 2.0  
**Link:** https://github.com/Arize-ai/phoenix

**Features:**
- LLM observability
- Embedding visualization
- Trace analysis
- Anomaly detection

**Integration:**
```python
import phoenix as px

px.launch_app()
# Automatically captures OpenAI, LangChain, LlamaIndex calls
```

**Best For:** Open source observability, embedding analysis

---

### 3. Helicone
**Type:** Open Source  
**License:** Apache 2.0  
**Link:** https://github.com/Helicone/helicone

**Features:**
- Open source LLM gateway
- Request/response logging
- Caching and rate limiting
- Analytics dashboard

**Integration:**
```python
# Point to Helicone proxy
client = OpenAI(
    api_key="your-key",
    base_url="https://oai.helicone.ai/v1"
)
```

**Best For:** Self-hosted monitoring, cost tracking

---

## Development Libraries

### 1. Microsoft Presidio
**Type:** Open Source  
**License:** MIT  
**Link:** https://github.com/microsoft/presidio

**Features:**
- PII detection and redaction
- Multiple entity types
- Custom recognizers
- Multi-language support

**Integration:**
```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Detect PII
results = analyzer.analyze(text=user_input, language="en")

# Redact PII
anonymized = anonymizer.anonymize(text=user_input, analyzer_results=results)
```

**Best For:** PII detection, data privacy compliance

---

### 2. Detoxify
**Type:** Open Source  
**License:** Apache 2.0  
**Link:** https://github.com/unitaryai/detoxify

**Features:**
- Toxic language detection
- Multiple toxicity categories
- Easy integration
- Pre-trained models

**Integration:**
```python
from detoxify import Detoxify

model = Detoxify('original')
results = model.predict("Your text here")
# Returns scores for toxicity, severe_toxicity, etc.
```

**Best For:** Content moderation, toxicity detection

---

### 3. TextBlob
**Type:** Open Source  
**License:** MIT  
**Link:** https://github.com/sloria/TextBlob

**Features:**
- Sentiment analysis
- Polarity scoring
- Simple API
- Good for basic filtering

**Integration:**
```python
from textblob import TextBlob

blob = TextBlob(user_input)
sentiment = blob.sentiment

if sentiment.polarity < -0.5:
    # Very negative sentiment
    pass
```

**Best For:** Basic sentiment analysis, simple filtering

---

## Commercial Solutions

### 1. Protect AI
**Link:** https://protectai.com/

**Features:**
- LLM firewall
- Real-time threat detection
- Compliance reporting
- Enterprise support

**Best For:** Enterprise deployments, compliance requirements

---

### 2. Hidden Layer
**Link:** https://hiddenlayer.com/

**Features:**
- AI security platform
- Model protection
- Threat detection
- Vulnerability management

**Best For:** Comprehensive AI security, model protection

---

### 3. Robust Intelligence
**Link:** https://robustintelligence.com/

**Features:**
- AI security testing
- Continuous validation
- Risk scoring
- Enterprise platform

**Best For:** Enterprise AI security, continuous testing

---

### 4. Calypso AI
**Link:** https://calypsoai.com/

**Features:**
- AI security testing
- Red team automation
- Compliance validation
- Training platform

**Best For:** Security testing, compliance validation

---

## Tool Comparison Matrix

| Tool | Type | License | Injection Detection | PII Detection | Output Filtering | Ease of Use |
|------|------|---------|---------------------|---------------|------------------|-------------|
| **Lakera Guard** | API | Commercial | ✅ Excellent | ✅ Excellent | ✅ Excellent | ⭐⭐⭐⭐⭐ |
| **Rebuff** | SDK | MIT | ✅ Good | ❌ | ⚠️ Basic | ⭐⭐⭐⭐ |
| **NeMo Guardrails** | Framework | Apache 2.0 | ✅ Good | ⚠️ Basic | ✅ Good | ⭐⭐⭐ |
| **Guardrails AI** | Framework | Apache 2.0 | ⚠️ Basic | ✅ Good | ✅ Excellent | ⭐⭐⭐⭐ |
| **Garak** | Scanner | Apache 2.0 | ✅ Testing | ⚠️ Testing | ❌ | ⭐⭐⭐ |
| **Presidio** | Library | MIT | ❌ | ✅ Excellent | ✅ Good | ⭐⭐⭐⭐ |
| **PyRIT** | Testing | MIT | ✅ Testing | ⚠️ Testing | ❌ | ⭐⭐⭐ |
| **LangSmith** | Monitoring | Commercial | ⚠️ Detection | ⚠️ Detection | ❌ | ⭐⭐⭐⭐⭐ |

**Legend:**
- ✅ Excellent/Yes
- ⚠️ Basic/Partial
- ❌ Not available

---

## Recommended Tool Stacks

### For Startups
```
Rebuff (Detection) + Guardrails AI (Output) + Helicone (Monitoring)
- Low cost
- Open source
- Easy integration
```

### For Enterprises
```
Lakera Guard (Detection) + NeMo Guardrails (Guardrails) + LangSmith (Monitoring)
- Comprehensive protection
- Enterprise support
- Compliance ready
```

### For Security Teams
```
Garak (Testing) + PyRIT (Red Team) + Protect AI (Production)
- Thorough testing
- Continuous validation
- Production protection
```

### For Research
```
PromptInject (Testing) + Garak (Scanning) + Custom Tools
- Flexible
- Extensible
- Research-focused
```

---

## Getting Started Guide

### Step 1: Assessment
```bash
# Run Garak to understand vulnerabilities
pip install garak
garak --model_type openai --model_name gpt-3.5-turbo
```

### Step 2: Basic Protection
```python
# Add Rebuff for injection detection
pip install rebuff

from rebuff import Rebuff
rb = Rebuff(api_key="your-key")
```

### Step 3: Output Validation
```python
# Add Guardrails AI for output validation
pip install guardrails-ai

import guardrails as gd
```

### Step 4: Monitoring
```python
# Set up LangSmith for monitoring
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your-key
```

### Step 5: Testing
```bash
# Regular security testing
garak --model_type openai --model_name gpt-3.5-turbo --probes prompt_injection
```

---

## Community Resources

### GitHub Discussions
- NeMo Guardrails: https://github.com/NVIDIA/NeMo-Guardrails/discussions
- Guardrails AI: https://github.com/guardrails-ai/guardrails/discussions

### Slack/Discord
- LangChain Discord: https://discord.gg/langchain
- AI Safety Slack: https://aisafety.slack.com

### Stack Overflow Tags
- `llm-security`
- `prompt-injection`
- `guardrails`

---

**Last Updated:** March 30, 2026  
**Maintained By:** AI-Mastery-2026 Security Team
