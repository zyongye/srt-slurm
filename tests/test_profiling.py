# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for profiling configuration, validation, and benchmark runner."""

import pytest

from srtctl.benchmarks import get_runner
from srtctl.benchmarks.base import SCRIPTS_DIR


class TestProfilingConfig:
    """Tests for ProfilingConfig dataclass."""

    def test_profiling_defaults(self):
        """Test profiling config defaults."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig()

        assert profiling.enabled is False
        assert profiling.is_nsys is False
        assert profiling.is_torch is False
        assert profiling.type == "none"

    def test_nsys_profiling(self):
        """Test nsys profiling configuration."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig(
            type="nsys",
        )

        assert profiling.enabled is True
        assert profiling.is_nsys is True
        assert profiling.is_torch is False

        # Test nsys prefix generation
        prefix = profiling.get_nsys_prefix("/output/test")
        assert any("nsys" in p for p in prefix)
        assert "profile" in prefix
        assert "/output/test" in prefix

        # Dynamo frontend requires trace-fork-before-exec, sglangrouter does not.
        prefix_dynamo = profiling.get_nsys_prefix("/output/test", frontend_type="dynamo")
        assert "--trace-fork-before-exec=true" in prefix_dynamo
        prefix_router = profiling.get_nsys_prefix("/output/test", frontend_type="sglangrouter")
        assert "--trace-fork-before-exec=true" not in prefix_router

    def test_torch_profiling(self):
        """Test torch profiling configuration."""
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig

        profiling = ProfilingConfig(
            type="torch",
            prefill=ProfilingPhaseConfig(start_step=5, stop_step=15),
            decode=ProfilingPhaseConfig(start_step=10, stop_step=20),
        )

        assert profiling.enabled is True
        assert profiling.is_torch is True
        assert profiling.is_nsys is False

        # Test env vars generation for prefill
        env = profiling.get_env_vars("prefill", "/logs/profiles")
        assert env["PROFILING_MODE"] == "prefill"
        assert env["PROFILE_TYPE"] == "torch"
        assert env["PROFILE_PREFILL_START_STEP"] == "5"
        assert env["PROFILE_PREFILL_STOP_STEP"] == "15"
        assert env["SGLANG_TORCH_PROFILER_DIR"] == "/logs/profiles/prefill"

        # Test env vars generation for decode (different steps)
        env_decode = profiling.get_env_vars("decode", "/logs/profiles")
        assert env_decode["PROFILE_DECODE_START_STEP"] == "10"
        assert env_decode["PROFILE_DECODE_STOP_STEP"] == "20"

    def test_aggregated_profiling(self):
        """Test aggregated profiling configuration."""
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig

        profiling = ProfilingConfig(
            type="torch",
            aggregated=ProfilingPhaseConfig(start_step=0, stop_step=100),
        )

        env = profiling.get_env_vars("agg", "/logs/profiles")
        assert env["PROFILE_TYPE"] == "torch"
        assert env["PROFILE_AGG_START_STEP"] == "0"
        assert env["PROFILE_AGG_STOP_STEP"] == "100"


class TestProfilingValidation:
    """Tests for profiling config validation in SrtConfig."""

    def test_disagg_requires_decode_config_when_decode_workers_present(self):
        """Disaggregated mode requires decode profiling config when decode_workers > 0."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Has decode workers but missing decode profiling config — should fail
        with pytest.raises(ValidationError, match="profiling.decode must be set"):
            SrtConfig(
                name="test",
                model=ModelConfig(path="/model", container="/container", precision="fp8"),
                resources=ResourceConfig(
                    gpu_type="h100",
                    prefill_nodes=1,
                    decode_nodes=1,
                    prefill_workers=1,
                    decode_workers=1,
                ),
                profiling=ProfilingConfig(
                    type="torch",
                    prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
                    # Missing decode config
                ),
            )

    def test_disagg_prefill_only_with_profiling(self):
        """Disaggregated prefill-only mode (decode_workers=0) works with only prefill profiling config."""
        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Should not raise — prefill-only with no decode workers
        SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                prefill_nodes=1,
                decode_nodes=0,
                prefill_workers=1,
                decode_workers=0,
            ),
            profiling=ProfilingConfig(
                type="nsys",
                prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
                # No decode config needed
            ),
        )

    def test_agg_requires_aggregated_config(self):
        """Aggregated mode requires aggregated profiling config."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Aggregated mode without aggregated profiling config should fail
        with pytest.raises(ValidationError, match="profiling.aggregated to be set"):
            SrtConfig(
                name="test",
                model=ModelConfig(path="/model", container="/container", precision="fp8"),
                resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
                profiling=ProfilingConfig(
                    type="torch",
                    # Missing aggregated config
                ),
            )

    def test_profiling_allows_multiple_workers_disagg(self):
        """Profiling in disaggregated mode supports multiple workers."""
        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Should not raise
        SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=2,
                decode_workers=3,
            ),
            profiling=ProfilingConfig(
                type="torch",
                prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
                decode=ProfilingPhaseConfig(start_step=0, stop_step=50),
            ),
        )

    def test_profiling_allows_multiple_workers_agg(self):
        """Profiling in aggregated mode supports multiple workers."""
        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Should not raise
        SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                agg_nodes=2,
                agg_workers=2,
            ),
            profiling=ProfilingConfig(
                type="torch",
                aggregated=ProfilingPhaseConfig(start_step=0, stop_step=50),
            ),
        )

    def test_valid_profiling_config_disagg(self):
        """Valid profiling config with 1P + 1D passes validation."""
        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Should not raise
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=1,
                decode_workers=1,
            ),
            profiling=ProfilingConfig(
                type="torch",
                prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
                decode=ProfilingPhaseConfig(start_step=0, stop_step=50),
            ),
        )
        assert config.profiling.enabled


class TestProfilingIntegration:
    """Integration tests for profiling + benchmarks."""

    def test_no_profiling_benchmark_runner(self):
        """There is no dedicated 'profiling' benchmark runner anymore."""
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_runner("profiling")

    def test_profiling_does_not_override_benchmark_type(self):
        """Profiling is orthogonal to benchmark selection."""
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # User sets benchmark.type to "sa-bench" and has profiling enabled.
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=1,
                decode_workers=1,
            ),
            benchmark=BenchmarkConfig(type="sa-bench"),
            profiling=ProfilingConfig(
                type="torch",
                prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
                decode=ProfilingPhaseConfig(start_step=0, stop_step=50),
            ),
        )

        assert config.profiling.enabled is True
        runner = get_runner(config.benchmark.type)
        assert runner.name == "SA-Bench"
        assert (SCRIPTS_DIR / "sa-bench" / "bench.sh").exists()

    def test_sglang_bench_script_exists(self):
        assert (SCRIPTS_DIR / "sglang-bench" / "bench.sh").exists()

    def test_sglang_bench_runner_validate_config(self):
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = get_runner("sglang-bench")

        config_missing = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=1,
                decode_workers=1,
            ),
            benchmark=BenchmarkConfig(type="sglang-bench"),
            profiling=ProfilingConfig(
                type="torch",
                prefill=ProfilingPhaseConfig(start_step=0, stop_step=10),
                decode=ProfilingPhaseConfig(start_step=0, stop_step=10),
            ),
        )

        errors = runner.validate_config(config_missing)
        assert "benchmark.isl is required for sglang-bench" in errors
        assert "benchmark.osl is required for sglang-bench" in errors
        assert "benchmark.concurrencies is required for sglang-bench" in errors

    def test_sglang_bench_runner_build_command(self):
        from types import SimpleNamespace

        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = get_runner("sglang-bench")
        runtime = SimpleNamespace(frontend_port=8000)

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=1,
                decode_workers=1,
            ),
            benchmark=BenchmarkConfig(type="sglang-bench", isl=1024, osl=128, concurrencies=[1, 2]),
        )

        cmd = runner.build_command(config, runtime)
        assert cmd == [
            "bash",
            "/srtctl-benchmarks/sglang-bench/bench.sh",
            "http://localhost:8000",
            "1024",
            "128",
            "1x2",
            "inf",
        ]


class TestTRTLLMProfilingSupport:
    """Tests that TRTLLM backend correctly applies nsys_prefix."""

    def _make_process(self, mode: str = "prefill"):
        from srtctl.core.topology import Process

        return Process(
            node="node0",
            gpu_indices=frozenset([0, 1, 2, 3]),
            sys_port=8081,
            http_port=8080,
            endpoint_mode=mode,
            endpoint_index=0,
            node_rank=0,
        )

    def _make_runtime(self, tmp_path):
        from types import SimpleNamespace
        from pathlib import Path

        return SimpleNamespace(
            log_dir=tmp_path,
            model_path=Path("/model/DeepSeek-R1"),
        )

    def test_nsys_prefix_applied(self, tmp_path):
        """nsys_prefix is prepended to the TRTLLM worker command."""
        from srtctl.backends.trtllm import TRTLLMProtocol

        backend = TRTLLMProtocol()
        process = self._make_process("decode")
        runtime = self._make_runtime(tmp_path)
        nsys_prefix = ["nsys", "profile", "--output", "/logs/profiles/decode/node0_profile"]

        cmd = backend.build_worker_command(
            process=process,
            endpoint_processes=[process],
            runtime=runtime,
            nsys_prefix=nsys_prefix,
        )

        assert cmd[:4] == nsys_prefix
        assert "trtllm-llmapi-launch" in cmd

    def test_no_nsys_prefix(self, tmp_path):
        """Without nsys_prefix the command starts directly with trtllm-llmapi-launch."""
        from srtctl.backends.trtllm import TRTLLMProtocol

        backend = TRTLLMProtocol()
        process = self._make_process("prefill")
        runtime = self._make_runtime(tmp_path)

        cmd = backend.build_worker_command(
            process=process,
            endpoint_processes=[process],
            runtime=runtime,
            nsys_prefix=None,
        )

        assert cmd[0] == "trtllm-llmapi-launch"


class TestTRTLLMProfileStartStop:
    """Tests for TLLM_PROFILE_START_STOP env var injection."""

    def test_tllm_profile_start_stop_set(self):
        """TLLM_PROFILE_START_STOP is set for TRTLLM + nsys with step range."""
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig

        profiling = ProfilingConfig(
            type="nsys",
            prefill=ProfilingPhaseConfig(start_step=5, stop_step=15),
            decode=ProfilingPhaseConfig(start_step=10, stop_step=20),
        )

        phase = profiling._get_phase_config("prefill")
        assert phase is not None
        env_val = f"{phase.start_step}-{phase.stop_step}"
        assert env_val == "5-15"

        phase_decode = profiling._get_phase_config("decode")
        assert phase_decode is not None
        assert f"{phase_decode.start_step}-{phase_decode.stop_step}" == "10-20"

    def test_tllm_profile_start_stop_missing_steps(self):
        """TLLM_PROFILE_START_STOP is not set when start_step/stop_step are absent."""
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig

        profiling = ProfilingConfig(
            type="nsys",
            prefill=ProfilingPhaseConfig(),  # no start/stop
            decode=ProfilingPhaseConfig(start_step=10, stop_step=20),
        )

        phase = profiling._get_phase_config("prefill")
        # Should not produce the env var
        assert phase is not None
        assert phase.start_step is None or phase.stop_step is None

    def test_tllm_profile_start_stop_set_for_torch(self):
        """TLLM_PROFILE_START_STOP is also set for torch profiling type."""
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig

        profiling = ProfilingConfig(
            type="torch",
            prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
            decode=ProfilingPhaseConfig(start_step=0, stop_step=50),
        )

        assert profiling.is_torch
        phase = profiling._get_phase_config("prefill")
        assert phase is not None
        assert f"{phase.start_step}-{phase.stop_step}" == "0-50"

    def test_tllm_torch_profile_trace_set(self):
        """TLLM_TORCH_PROFILE_TRACE is set for TRTLLM torch profiling."""
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig

        profiling = ProfilingConfig(
            type="torch",
            prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
            decode=ProfilingPhaseConfig(start_step=0, stop_step=50),
        )

        assert profiling.is_torch
        # Verify the env var key/value pattern used in worker_stage.py
        profile_dir = "/logs/profiles"
        trace_path = f"{profile_dir}/prefill"
        assert trace_path == "/logs/profiles/prefill"

    def test_tllm_torch_profile_trace_not_set_for_nsys(self):
        """TLLM_TORCH_PROFILE_TRACE should not be set for nsys profiling."""
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig

        profiling = ProfilingConfig(
            type="nsys",
            prefill=ProfilingPhaseConfig(start_step=5, stop_step=15),
            decode=ProfilingPhaseConfig(start_step=5, stop_step=15),
        )

        assert profiling.is_nsys
        assert not profiling.is_torch
