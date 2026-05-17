"""Tests for llmstack download utilities."""

from __future__ import annotations


class TestIterDownloadTargets:
    """Tests for iter_download_targets()."""

    def test_empty_for_no_tiers(self):
        # Create a temporary models.ini file
        import tempfile
        from pathlib import Path

        from llmstack.tiers import iter_download_targets

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".ini",
            delete=False,
        ) as f:
            f.write("[code-fast]\n")
            f.write("hf_repo = test/repo\n")
            f.write("hf_file = model.gguf\n")
            f.write("role = agent\n")
            f.write("description = Test\n")
            f.write("ctx_size = 128000\n")
            path = Path(f.name)

        try:
            result = list(iter_download_targets(path))
            assert len(result) == 1
            assert result[0].tier == "code-fast"
        finally:
            path.unlink()

    def test_skips_bedrock_tiers(self):
        import tempfile
        from pathlib import Path

        from llmstack.tiers import iter_download_targets

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".ini",
            delete=False,
        ) as f:
            f.write("[plan]\n")
            f.write("backend = litellm\n")
            f.write("model = bedrock/eu.anthropic.claude-3-5-sonnet\n")
            f.write("role = plan\n")
            f.write("description = Plan mode\n")
            f.write("ctx_size = 200000\n")
            path = Path(f.name)

        try:
            result = list(iter_download_targets(path))
            assert len(result) == 0
        finally:
            path.unlink()
