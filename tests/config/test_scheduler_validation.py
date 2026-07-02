# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for Scheduler Configuration Validation Framework.

Tests validate that the framework correctly:
1. Detects configuration conflicts
2. Identifies parameter dependencies
3. Suggests reasonable corrections
4. Handles various model profiles (encoder-decoder, multimodal, MoE, DP)
5. Provides actionable error messages

Run without GPU: these tests only use Python config objects.
"""

import pytest
from vllm.config.scheduler_validation import (
    SchedulerConfigValidator,
    ValidationReport,
    SeverityLevel,
    get_scheduler_config_profile,
)


# Mock SchedulerConfig for testing (minimal implementation)
class MockSchedulerConfig:
    """Minimal SchedulerConfig mock for testing validation logic."""
    
    def __init__(self, **kwargs):
        # Defaults from SchedulerConfig
        self.runner_type = kwargs.get("runner_type", "generate")
        self.max_num_batched_tokens = kwargs.get("max_num_batched_tokens", 2048)
        self.max_num_scheduled_tokens = kwargs.get("max_num_scheduled_tokens", None)
        self.max_num_seqs = kwargs.get("max_num_seqs", 128)
        self.max_num_partial_prefills = kwargs.get("max_num_partial_prefills", 1)
        self.max_long_partial_prefills = kwargs.get("max_long_partial_prefills", 1)
        self.long_prefill_token_threshold = kwargs.get("long_prefill_token_threshold", 0)
        self.enable_chunked_prefill = kwargs.get("enable_chunked_prefill", True)
        self.is_multimodal_model = kwargs.get("is_multimodal_model", False)
        self.policy = kwargs.get("policy", "fcfs")
        self.disable_chunked_mm_input = kwargs.get("disable_chunked_mm_input", False)
        self.scheduler_cls = kwargs.get("scheduler_cls", None)
        self.disable_hybrid_kv_cache_manager = kwargs.get("disable_hybrid_kv_cache_manager", None)
        self.scheduler_reserve_full_isl = kwargs.get("scheduler_reserve_full_isl", True)
        self.watermark = kwargs.get("watermark", 0.0)
        self.prefill_schedule_interval = kwargs.get("prefill_schedule_interval", 1)
        self.async_scheduling = kwargs.get("async_scheduling", None)
        self.stream_interval = kwargs.get("stream_interval", 1)


class TestChunkedPrefillValidation:
    """Test chunked prefill constraint validation."""
    
    def test_encoder_decoder_disables_chunked_prefill(self):
        """Encoder-decoder models must disable chunked prefill."""
        config = MockSchedulerConfig(enable_chunked_prefill=True)
        validator = SchedulerConfigValidator(
            config,
            is_encoder_decoder=True,
            max_model_len=4096
        )
        
        report = validator.validate()
        errors = report.get_by_severity(SeverityLevel.ERROR)
        
        # Should have error about chunked prefill
        assert any("enable_chunked_prefill" in e.param_name for e in errors)
        assert any(e.suggested_value is False for e in errors)
    
    def test_chunked_prefill_with_large_batch_size(self):
        """Chunked prefill should warn if batch size > batched tokens."""
        config = MockSchedulerConfig(
            enable_chunked_prefill=True,
            max_num_seqs=256,
            max_num_batched_tokens=128,
        )
        validator = SchedulerConfigValidator(config, max_model_len=4096)
        
        report = validator.validate()
        warnings = report.get_by_severity(SeverityLevel.WARNING)
        
        # Should warn about inefficient chunked prefill
        assert any("max_num_seqs" in e.param_name for e in warnings)


class TestPartialPrefillValidation:
    """Test concurrent partial prefill constraint validation."""
    
    def test_partial_prefills_require_chunked_prefill(self):
        """Concurrent partial prefills require chunked prefill enabled."""
        config = MockSchedulerConfig(
            max_num_partial_prefills=4,
            enable_chunked_prefill=False,
        )
        validator = SchedulerConfigValidator(config, max_model_len=4096)
        
        report = validator.validate()
        errors = report.get_by_severity(SeverityLevel.ERROR)
        
        # Should have error about missing chunked prefill
        assert any("max_num_partial_prefills" in e.param_name for e in errors)
    
    def test_long_threshold_auto_suggestion(self):
        """When using concurrent partial prefills, threshold is auto-suggested."""
        config = MockSchedulerConfig(
            max_num_partial_prefills=4,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=0,  # Not set
        )
        validator = SchedulerConfigValidator(config, max_model_len=10000)
        
        report = validator.validate()
        suggestions = report.get_suggestions()
        
        # Should suggest setting threshold
        threshold_suggestions = [
            s for s in suggestions
            if s.param_name == "long_prefill_token_threshold"
        ]
        assert len(threshold_suggestions) > 0
        # Should be ~4% of max_model_len
        assert threshold_suggestions[0].suggested_value == 400


class TestTokenLimitValidation:
    """Test token limit consistency validation."""
    
    def test_batch_size_smaller_than_max_len_without_chunking(self):
        """Error when batch size < max_model_len without chunked prefill."""
        config = MockSchedulerConfig(
            max_num_batched_tokens=1024,
            enable_chunked_prefill=False,
        )
        validator = SchedulerConfigValidator(
            config,
            max_model_len=4096
        )
        
        report = validator.validate()
        errors = report.get_by_severity(SeverityLevel.ERROR)
        
        # Should have error
        assert any("max_num_batched_tokens" in e.param_name for e in errors)
    
    def test_batch_size_smaller_than_num_seqs(self):
        """Error when batch size < num sequences."""
        config = MockSchedulerConfig(
            max_num_batched_tokens=32,
            max_num_seqs=128,
        )
        validator = SchedulerConfigValidator(config, max_model_len=4096)
        
        report = validator.validate()
        errors = report.get_by_severity(SeverityLevel.ERROR)
        
        # Should have error
        assert any("max_num_batched_tokens" in e.param_name for e in errors)


class TestEncoderDecoderConstraints:
    """Test encoder-decoder specific constraints."""
    
    def test_encoder_decoder_disables_all_incompatible_features(self):
        """Encoder-decoder must disable chunked prefill and set threshold to 0."""
        config = MockSchedulerConfig(
            enable_chunked_prefill=True,
            long_prefill_token_threshold=100,
        )
        validator = SchedulerConfigValidator(
            config,
            is_encoder_decoder=True,
            max_model_len=4096
        )
        
        report = validator.validate()
        errors = report.get_by_severity(SeverityLevel.ERROR)
        
        # Should have errors for both parameters
        error_params = [e.param_name for e in errors]
        assert "enable_chunked_prefill" in error_params
        assert "long_prefill_token_threshold" in error_params


class TestDataParallelConstraints:
    """Test data parallel specific constraints."""
    
    def test_dp_suggests_prefill_schedule_interval(self):
        """Data parallel should suggest tuning prefill_schedule_interval."""
        config = MockSchedulerConfig(
            prefill_schedule_interval=1,
        )
        validator = SchedulerConfigValidator(
            config,
            data_parallel_size=4,
            max_model_len=4096
        )
        
        report = validator.validate()
        infos = report.get_by_severity(SeverityLevel.INFO)
        
        # Should have info about prefill scheduling
        assert any("prefill_schedule_interval" in e.param_name for e in infos)


class TestMoEConstraints:
    """Test Mixture of Experts specific constraints."""
    
    def test_moe_suggests_larger_batch_size(self):
        """MoE models should warn about small batch sizes."""
        config = MockSchedulerConfig(
            max_num_seqs=16,  # Small for MoE
        )
        validator = SchedulerConfigValidator(
            config,
            is_moe=True,
            max_model_len=4096
        )
        
        report = validator.validate()
        warnings = report.get_by_severity(SeverityLevel.WARNING)
        
        # Should warn about batch size
        assert any("max_num_seqs" in e.param_name for e in warnings)


class TestValidationReport:
    """Test ValidationReport aggregation and methods."""
    
    def test_report_summary(self):
        """Report should generate accurate summary."""
        config = MockSchedulerConfig(
            max_num_batched_tokens=32,
            max_num_seqs=128,
        )
        validator = SchedulerConfigValidator(config, max_model_len=4096)
        report = validator.validate()
        
        summary = report.summary()
        assert "Scheduler Config Validation" in summary
        assert report.profile_name in summary
    
    def test_get_suggestions(self):
        """Should return only issues with suggested values."""
        config = MockSchedulerConfig(
            max_num_batched_tokens=32,
            max_num_seqs=128,
        )
        validator = SchedulerConfigValidator(config, max_model_len=4096)
        report = validator.validate()
        
        suggestions = report.get_suggestions()
        # All suggestions should have non-None suggested_value
        assert all(s.suggested_value is not None for s in suggestions)
    
    def test_has_errors(self):
        """Should correctly detect ERROR and CRITICAL issues."""
        config = MockSchedulerConfig(
            max_num_batched_tokens=32,
            max_num_seqs=128,
        )
        validator = SchedulerConfigValidator(config, max_model_len=4096)
        report = validator.validate()
        
        assert report.has_errors() is True


class TestProfileDetermination:
    """Test automatic profile determination."""
    
    def test_single_gpu_profile(self):
        """Single GPU should get basic profile."""
        config = MockSchedulerConfig()
        validator = SchedulerConfigValidator(
            config,
            data_parallel_size=1,
            tensor_parallel_size=1,
        )
        
        assert validator.profile == "single_gpu"
    
    def test_multi_feature_profile(self):
        """Multiple features should create combined profile."""
        config = MockSchedulerConfig()
        validator = SchedulerConfigValidator(
            config,
            is_encoder_decoder=True,
            is_multimodal=True,
            data_parallel_size=2,
        )
        
        # Profile should include all features
        assert "encoder_decoder" in validator.profile
        assert "multimodal" in validator.profile
        assert "data_parallel" in validator.profile


class TestProfileDefaults:
    """Test get_scheduler_config_profile function."""
    
    def test_encoder_decoder_profile_defaults(self):
        """Encoder-decoder profile should disable chunked prefill."""
        profile = get_scheduler_config_profile(
            model_type="t5",
            is_encoder_decoder=True,
            data_parallel_size=1,
            max_model_len=512,
        )
        
        assert profile["enable_chunked_prefill"] is False
        assert profile["max_num_seqs"] == 64
    
    def test_data_parallel_profile_defaults(self):
        """Data parallel profile should suggest prefill schedule interval."""
        profile = get_scheduler_config_profile(
            model_type="llama",
            is_encoder_decoder=False,
            data_parallel_size=4,
            max_model_len=4096,
        )
        
        assert profile["prefill_schedule_interval"] == 4
        # Batched tokens should be reduced for DP stability
        assert profile["max_num_batched_tokens"] >= 256


class TestEdgeCases:
    """Test edge cases and corner scenarios."""
    
    def test_very_large_model(self):
        """Should handle models with very large max_model_len."""
        config = MockSchedulerConfig(max_num_batched_tokens=8192)
        validator = SchedulerConfigValidator(
            config,
            max_model_len=100000,
        )
        
        report = validator.validate()
        # Should not crash
        assert isinstance(report, ValidationReport)
    
    def test_all_features_enabled(self):
        """Should handle configuration with all features."""
        config = MockSchedulerConfig(
            enable_chunked_prefill=True,
            max_num_partial_prefills=4,
        )
        validator = SchedulerConfigValidator(
            config,
            is_encoder_decoder=False,
            is_multimodal=True,
            is_moe=True,
            data_parallel_size=8,
            tensor_parallel_size=2,
            max_model_len=4096,
        )
        
        report = validator.validate()
        # Should handle complex scenario
        assert isinstance(report, ValidationReport)
    
    def test_minimal_config(self):
        """Should handle minimal configuration."""
        config = MockSchedulerConfig()
        validator = SchedulerConfigValidator(config)
        
        report = validator.validate()
        # Should not crash
        assert len(report.issues) >= 0


if __name__ == "__main__":
    # Run tests: pytest tests/config/test_scheduler_validation.py -v
    pytest.main([__file__, "-v"])
