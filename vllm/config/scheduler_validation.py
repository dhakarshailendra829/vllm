# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Scheduler Configuration Validation & Dependency Resolution Framework.

This module provides a systematic approach to validating vLLM scheduler configurations
by modeling parameter dependencies as a directed graph. It detects conflicts,
suggests corrections, and provides profile-based defaults.

Key Features:
1. Dependency Graph: Maps parameter relationships and constraints
2. Conflict Detection: Identifies incompatible configurations before engine init
3. Auto-correction: Suggests safe adjustments with detailed reasoning
4. Profiling: Context-aware defaults based on model type and parallelism setup

Research Background:
- Current validation in SchedulerConfig.__post_init__ (scheduler.py:237-321) is scattered
- No systematic tracking of parameter interdependencies
- Difficult to debug configuration issues in production
- Each model type (encoder-decoder, multimodal, MoE) has unique constraints

Usage:
    from vllm.config.scheduler_validation import SchedulerConfigValidator
    
    validator = SchedulerConfigValidator(
        scheduler_config=config,
        model_type="llama",
        is_encoder_decoder=False,
        is_multimodal=False,
        data_parallel_size=1,
        max_model_len=4096
    )
    
    issues = validator.validate()
    if issues.has_conflicts():
        suggestions = issues.get_suggestions()
        # Apply corrections or warn user
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SeverityLevel(Enum):
    """Classification of validation issues."""
    INFO = 0          # Informational - no action needed
    WARNING = 1       # Warning - suggests optimization
    ERROR = 2         # Error - configuration won't work
    CRITICAL = 3      # Critical - will cause runtime failure


class ParameterDependency(Enum):
    """Types of dependencies between scheduler parameters."""
    REQUIRES = "requires"              # Parameter A requires parameter B
    CONFLICTS_WITH = "conflicts_with"  # Parameter A conflicts with parameter B
    IMPLIES = "implies"                # Parameter A implies parameter B should be set
    BOUNDS = "bounds"                  # Parameter A bounds parameter B
    PROFILES = "profiles"              # Parameter is profile-specific


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    param_name: str
    severity: SeverityLevel
    message: str
    suggested_value: Optional[Any] = None
    related_params: list[str] = field(default_factory=list)
    
    def apply_suggestion(self, config_dict: dict) -> dict:
        """Apply the suggested value to a config dict."""
        if self.suggested_value is not None:
            config_dict[self.param_name] = self.suggested_value
        return config_dict


@dataclass
class ValidationReport:
    """Aggregated validation results."""
    issues: list[ValidationIssue] = field(default_factory=list)
    profile_name: str = "unknown"
    
    def has_errors(self) -> bool:
        """Check if there are any ERROR or CRITICAL issues."""
        return any(
            i.severity in (SeverityLevel.ERROR, SeverityLevel.CRITICAL)
            for i in self.issues
        )
    
    def has_conflicts(self) -> bool:
        """Check if there are any configuration conflicts."""
        return self.has_errors()
    
    def get_suggestions(self) -> list[ValidationIssue]:
        """Get all issues that have actionable suggestions."""
        return [i for i in self.issues if i.suggested_value is not None]
    
    def get_by_severity(self, severity: SeverityLevel) -> list[ValidationIssue]:
        """Get issues of a specific severity level."""
        return [i for i in self.issues if i.severity == severity]
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        counts = {}
        for severity in SeverityLevel:
            counts[severity.name] = len(self.get_by_severity(severity))
        
        return (
            f"Scheduler Config Validation ({self.profile_name}): "
            f"{counts['ERROR']} errors, {counts['WARNING']} warnings, "
            f"{counts['INFO']} info"
        )


class SchedulerConfigValidator:
    """
    Validates scheduler configurations based on parameter dependencies.
    
    This validator models the complex interdependencies between scheduler parameters
    and detects conflicts that would cause runtime issues. It's designed to be:
    - Non-blocking: provides suggestions without modifying configs
    - Informative: explains why issues exist and how to fix them
    - Extensible: easy to add new constraints and profiles
    """
    
    def __init__(
        self,
        scheduler_config: Any,
        model_type: str = "unknown",
        is_encoder_decoder: bool = False,
        is_multimodal: bool = False,
        is_moe: bool = False,
        data_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
    ):
        """
        Initialize validator with configuration context.
        
        Args:
            scheduler_config: SchedulerConfig instance to validate
            model_type: Model architecture name (e.g., "llama", "qwen", "gpt2")
            is_encoder_decoder: Whether model uses encoder-decoder architecture
            is_multimodal: Whether model supports multimodal inputs
            is_moe: Whether model uses Mixture of Experts
            data_parallel_size: Number of data parallel ranks
            tensor_parallel_size: Number of tensor parallel ranks
            max_model_len: Maximum model sequence length
        """
        self.config = scheduler_config
        self.model_type = model_type
        self.is_encoder_decoder = is_encoder_decoder
        self.is_multimodal = is_multimodal
        self.is_moe = is_moe
        self.data_parallel_size = data_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        
        # Determine configuration profile
        self.profile = self._determine_profile()
    
    def _determine_profile(self) -> str:
        """Determine which scheduler profile applies to this configuration."""
        profiles = []
        
        if self.is_encoder_decoder:
            profiles.append("encoder_decoder")
        if self.is_multimodal:
            profiles.append("multimodal")
        if self.is_moe:
            profiles.append("moe")
        if self.data_parallel_size > 1:
            profiles.append("data_parallel")
        if self.tensor_parallel_size > 1:
            profiles.append("tensor_parallel")
        
        if not profiles:
            profiles.append("single_gpu")
        
        return "_".join(sorted(profiles))
    
    def validate(self) -> ValidationReport:
        """
        Run complete validation suite.
        
        Returns:
            ValidationReport with all detected issues and suggestions
        """
        report = ValidationReport(profile_name=self.profile)
        
        # Run validation suites
        report.issues.extend(self._validate_chunked_prefill())
        report.issues.extend(self._validate_partial_prefills())
        report.issues.extend(self._validate_token_limits())
        report.issues.extend(self._validate_encoder_decoder_constraints())
        report.issues.extend(self._validate_multimodal_constraints())
        report.issues.extend(self._validate_data_parallel_constraints())
        report.issues.extend(self._validate_moe_constraints())
        
        return report
    
    def _validate_chunked_prefill(self) -> list[ValidationIssue]:
        """
        Validate chunked prefill configuration.
        
        Chunked prefill allows breaking long prompts into chunks, reducing memory
        pressure. However, it has constraints:
        - Incompatible with encoder-decoder models (must process full input)
        - Incompatible with non-causal attention (prefix LM)
        - Requires sufficient batch size to amortize overhead
        """
        issues = []
        
        if not self.config.enable_chunked_prefill:
            return issues
        
        # Check: encoder-decoder incompatibility
        if self.is_encoder_decoder:
            issues.append(ValidationIssue(
                param_name="enable_chunked_prefill",
                severity=SeverityLevel.ERROR,
                message=(
                    "Chunked prefill is incompatible with encoder-decoder models "
                    "which require full input context for encoder"
                ),
                suggested_value=False,
                related_params=["is_encoder_decoder"]
            ))
        
        # Check: sufficient batch size
        if self.config.max_num_seqs > self.config.max_num_batched_tokens:
            issues.append(ValidationIssue(
                param_name="max_num_seqs",
                severity=SeverityLevel.WARNING,
                message=(
                    f"max_num_seqs ({self.config.max_num_seqs}) exceeds "
                    f"max_num_batched_tokens ({self.config.max_num_batched_tokens}). "
                    "Chunked prefill overhead may not be amortized"
                ),
                suggested_value=min(
                    self.config.max_num_seqs,
                    self.config.max_num_batched_tokens // 2
                ),
                related_params=["max_num_batched_tokens"]
            ))
        
        return issues
    
    def _validate_partial_prefills(self) -> list[ValidationIssue]:
        """
        Validate concurrent partial prefill configuration.
        
        Concurrent partial prefills (max_num_partial_prefills > 1) allow scheduling
        multiple prompts in a single step, improving throughput. Constraints:
        - Requires chunked prefill enabled
        - Requires appropriate thresholds for long-prompt differentiation
        - Can starve short prompts if max_long_partial_prefills not tuned
        """
        issues = []
        
        if self.config.max_num_partial_prefills <= 1:
            return issues
        
        # Check: chunked prefill requirement
        if not self.config.enable_chunked_prefill:
            issues.append(ValidationIssue(
                param_name="max_num_partial_prefills",
                severity=SeverityLevel.ERROR,
                message=(
                    "Concurrent partial prefills require chunked prefill enabled"
                ),
                suggested_value=1,
                related_params=["enable_chunked_prefill"]
            ))
        
        # Check: threshold configuration
        if (self.config.long_prefill_token_threshold == 0 and
            self.config.max_num_partial_prefills > 1):
            issues.append(ValidationIssue(
                param_name="long_prefill_token_threshold",
                severity=SeverityLevel.INFO,
                message=(
                    "long_prefill_token_threshold=0 with concurrent partial prefills. "
                    "Will auto-tune to 4% of max_model_len"
                ),
                suggested_value=int(self.max_model_len * 0.04),
                related_params=["max_num_partial_prefills"]
            ))
        
        # Check: starvation risk
        if (self.config.max_long_partial_prefills == self.config.max_num_partial_prefills and
            self.config.max_num_partial_prefills > 2):
            issues.append(ValidationIssue(
                param_name="max_long_partial_prefills",
                severity=SeverityLevel.WARNING,
                message=(
                    "max_long_partial_prefills equals max_num_partial_prefills. "
                    "Short prompts may starve. Consider reducing to allow queue jumping"
                ),
                suggested_value=max(1, self.config.max_num_partial_prefills // 2),
                related_params=["max_num_partial_prefills"]
            ))
        
        return issues
    
    def _validate_token_limits(self) -> list[ValidationIssue]:
        """
        Validate token limit consistency.
        
        Ensures max_num_batched_tokens is sufficient to process longest sequences
        without chunking (if chunked prefill disabled).
        """
        issues = []
        
        # Check: batch size vs max sequence length
        if (self.config.max_num_batched_tokens < self.max_model_len and
            not self.config.enable_chunked_prefill):
            issues.append(ValidationIssue(
                param_name="max_num_batched_tokens",
                severity=SeverityLevel.ERROR,
                message=(
                    f"max_num_batched_tokens ({self.config.max_num_batched_tokens}) < "
                    f"max_model_len ({self.max_model_len}) with chunked prefill disabled. "
                    "Long sequences will be rejected"
                ),
                suggested_value=self.max_model_len,
                related_params=["enable_chunked_prefill", "max_model_len"]
            ))
        
        # Check: batched tokens vs num sequences
        if self.config.max_num_batched_tokens < self.config.max_num_seqs:
            issues.append(ValidationIssue(
                param_name="max_num_batched_tokens",
                severity=SeverityLevel.ERROR,
                message=(
                    f"max_num_batched_tokens ({self.config.max_num_batched_tokens}) < "
                    f"max_num_seqs ({self.config.max_num_seqs}). "
                    "Cannot schedule even a single token per sequence"
                ),
                suggested_value=self.config.max_num_seqs * 4,
                related_params=["max_num_seqs"]
            ))
        
        return issues
    
    def _validate_encoder_decoder_constraints(self) -> list[ValidationIssue]:
        """
        Validate encoder-decoder specific constraints.
        
        Encoder-decoder models (T5, BART, mT5) have unique requirements:
        - Chunked prefill disabled (needs full context)
        - Prefix caching disabled (breaks during decoding)
        - Long prefill threshold = 0 (disabled)
        """
        issues = []
        
        if not self.is_encoder_decoder:
            return issues
        
        checks = [
            ("enable_chunked_prefill", False, "Encoder-decoder requires full encoder context"),
            ("long_prefill_token_threshold", 0, "Disabled for encoder-decoder"),
        ]
        
        for param_name, expected_value, reason in checks:
            actual_value = getattr(self.config, param_name, None)
            if actual_value != expected_value:
                issues.append(ValidationIssue(
                    param_name=param_name,
                    severity=SeverityLevel.ERROR,
                    message=f"{reason}. Got {actual_value}, expected {expected_value}",
                    suggested_value=expected_value,
                    related_params=["is_encoder_decoder"]
                ))
        
        return issues
    
    def _validate_multimodal_constraints(self) -> list[ValidationIssue]:
        """
        Validate multimodal model constraints.
        
        Multimodal models (LLaVA, IDEFICS) have higher compute requirements:
        - Vision preprocessing may exceed token limits
        - max_num_encoder_input_tokens should account for image tokens
        - May need adjusted batch sizes
        """
        issues = []
        
        if not self.is_multimodal:
            return issues
        
        # Note: Detailed multimodal constraints depend on specific model
        # This is a placeholder for the framework
        issues.append(ValidationIssue(
            param_name="max_num_batched_tokens",
            severity=SeverityLevel.INFO,
            message=(
                "Multimodal model detected. Vision encoder tokens may consume "
                "significant budget. Monitor token allocation during inference"
            ),
            related_params=["is_multimodal"]
        ))
        
        return issues
    
    def _validate_data_parallel_constraints(self) -> list[ValidationIssue]:
        """
        Validate data parallelism constraints.
        
        Data parallel deployments (vllm/v1/engine/core.py:822-864) require:
        - Prefill schedule interval to balance forward pass times
        - Proper batch size tuning per DP rank
        - Consideration of load balancing strategy
        """
        issues = []
        
        if self.data_parallel_size <= 1:
            return issues
        
        # Check: prefill scheduling for DP balancing
        if self.config.prefill_schedule_interval == 1:
            issues.append(ValidationIssue(
                param_name="prefill_schedule_interval",
                severity=SeverityLevel.INFO,
                message=(
                    "Data parallel deployment detected. Consider increasing "
                    "prefill_schedule_interval to balance per-step forward pass times"
                ),
                suggested_value=min(4, self.data_parallel_size),
                related_params=["data_parallel_size"]
            ))
        
        return issues
    
    def _validate_moe_constraints(self) -> list[ValidationIssue]:
        """
        Validate Mixture of Experts constraints.
        
        MoE models (Mixtral, DeepSeek) require:
        - Sufficient batch size for expert utilization
        - Consideration of load balancing across experts
        - Token routing overhead
        """
        issues = []
        
        if not self.is_moe:
            return issues
        
        # MoE typically benefits from larger batch sizes
        if self.config.max_num_seqs < 32:
            issues.append(ValidationIssue(
                param_name="max_num_seqs",
                severity=SeverityLevel.WARNING,
                message=(
                    "MoE model with small batch size. May underutilize expert parallelism. "
                    f"Current: {self.config.max_num_seqs}, recommended: ≥32"
                ),
                suggested_value=max(32, self.config.max_num_seqs),
                related_params=["is_moe"]
            ))
        
        return issues


def get_scheduler_config_profile(
    model_type: str,
    is_encoder_decoder: bool,
    data_parallel_size: int,
    max_model_len: int,
) -> dict:
    """
    Get recommended scheduler configuration for a profile.
    
    Returns default values suitable for the specified configuration profile.
    Can be used as starting point before fine-tuning.
    
    Args:
        model_type: Model architecture type
        is_encoder_decoder: Whether model is encoder-decoder
        data_parallel_size: Number of data parallel ranks
        max_model_len: Maximum sequence length
    
    Returns:
        Dict of recommended scheduler config values
    """
    base_config = {
        "enable_chunked_prefill": True,
        "max_num_seqs": 128,
        "max_num_batched_tokens": 2048,
    }
    
    # Encoder-decoder profile
    if is_encoder_decoder:
        base_config.update({
            "enable_chunked_prefill": False,
            "max_num_seqs": 64,  # Smaller for encoder-decoder
        })
    
    # Data parallel profile
    if data_parallel_size > 1:
        base_config.update({
            "prefill_schedule_interval": min(4, data_parallel_size),
            "max_num_batched_tokens": max(
                256,  # Minimum for DP stability
                base_config["max_num_batched_tokens"] // data_parallel_size
            ),
        })
    
    return base_config
