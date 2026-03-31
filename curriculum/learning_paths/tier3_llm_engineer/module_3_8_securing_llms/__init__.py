"""
Module 3.8: Securing LLMs

Production-ready LLM security implementations:
- Prompt Hacking: Injection detection, jailbreak prevention
- Backdoors: Poisoning detection, trigger detection
- Defense: Input sanitization, output filtering, rate limiting
- Red Teaming: Automated testing, vulnerability scanning
"""

from .prompt_hacking import (
    PromptSecurityAnalyzer,
    InjectionDetector,
    JailbreakDetector,
    PromptLeakagePreventer,
    SecurityConfig,
)
from .backdoors import (
    BackdoorDetector,
    PoisoningDetector,
    TriggerDetector,
    ModelInspector,
    BackdoorConfig,
)
from .defense import (
    DefenseLayer,
    InputSanitizer,
    OutputFilter,
    RateLimiter,
    AccessControl,
    DefenseConfig,
)
from .red_teaming import (
    RedTeamFramework,
    AutomatedRedTeamer,
    GarakIntegration,
    VulnerabilityScanner,
    OWASPChecker,
    RedTeamConfig,
)

__all__ = [
    # Prompt Hacking
    "PromptSecurityAnalyzer",
    "InjectionDetector",
    "JailbreakDetector",
    "PromptLeakagePreventer",
    "SecurityConfig",
    # Backdoors
    "BackdoorDetector",
    "PoisoningDetector",
    "TriggerDetector",
    "ModelInspector",
    "BackdoorConfig",
    # Defense
    "DefenseLayer",
    "InputSanitizer",
    "OutputFilter",
    "RateLimiter",
    "AccessControl",
    "DefenseConfig",
    # Red Teaming
    "RedTeamFramework",
    "AutomatedRedTeamer",
    "GarakIntegration",
    "VulnerabilityScanner",
    "OWASPChecker",
    "RedTeamConfig",
]

__version__ = "1.0.0"
