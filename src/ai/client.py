"""AI client abstraction supporting multiple providers."""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from google import genai
from google.genai import types

from ..models import AIConfig, AIProvider
from .tokens import record_usage


class AIClient(ABC):
    """Abstract base class for AI clients."""

    @abstractmethod
    async def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion from AI model.

        Args:
            system: System prompt
            user: User prompt
            temperature: Optional sampling temperature override
            max_tokens: Optional maximum tokens override

        Returns:
            str: Generated completion text
        """
        pass


class _LocalTempDir:
    """Small temp-dir helper that keeps Codex artifacts out of the project tree."""

    def __init__(self, prefix: str):
        self._temp_dir = tempfile.TemporaryDirectory(
            prefix=f"{prefix}{uuid.uuid4().hex}-",
            ignore_cleanup_errors=True,
        )
        self.name = self._temp_dir.name

    def cleanup(self) -> None:
        try:
            self._temp_dir.cleanup()
        except OSError:
            pass


class CodexCliClient(AIClient):
    """Client that uses a locally authenticated Codex CLI session."""

    _DEFAULT_REASONING_CONFIG = 'model_reasoning_effort="medium"'
    _REASONING_CONFIG_KEY = "model_reasoning_effort"
    _PURE_TEXT_SUFFIX = """You are being used as a pure text completion backend.
Do not edit files.
Do not run shell commands.
Do not inspect the repository.
Do not use tools.
Return only the requested output."""

    def __init__(self, config: AIConfig):
        self.config = config
        self.command = self._resolve_command(config.codex_command)
        self.timeout_sec = config.codex_timeout_sec
        self.use_output_last_message = config.codex_use_output_last_message
        self.use_json = config.codex_use_json
        self.extra_args = list(config.codex_extra_args)
        self._supported_flags: Optional[set[str]] = None

    @staticmethod
    def _resolve_command(command: str) -> str:
        path = Path(command).expanduser()
        if path.is_absolute() or path.parent != Path("."):
            return str(path)
        return shutil.which(command) or command

    async def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a completion by invoking `codex exec`."""
        prompt = self._build_prompt(system=system, user=user)
        return await asyncio.to_thread(self._run_codex_exec, prompt)

    def _build_prompt(self, system: str, user: str) -> str:
        return f"SYSTEM:\n{system}\n\nUSER:\n{user}\n\n{self._PURE_TEXT_SUFFIX}"

    def _get_supported_flags(self) -> set[str]:
        if self._supported_flags is not None:
            return self._supported_flags

        try:
            result = subprocess.run(
                [self.command, "exec", "--help"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            self._supported_flags = set()
            return self._supported_flags

        help_text = f"{result.stdout}\n{result.stderr}"
        self._supported_flags = {
            flag
            for flag in (
                "--json",
                "--output-last-message",
                "--color",
                "--sandbox",
                "--skip-git-repo-check",
                "-c",
                "--config",
            )
            if flag in help_text
        }
        return self._supported_flags

    @staticmethod
    def _args_include_flag(args: list[str], flag: str) -> bool:
        return any(arg == flag for arg in args)

    @classmethod
    def _args_configure_reasoning(cls, args: list[str]) -> bool:
        return any(cls._REASONING_CONFIG_KEY in arg for arg in args)

    def _build_args(self, output_path: Optional[str], flags: set[str]) -> list[str]:
        args = [self.command, "exec"]
        extra_args = list(self.extra_args)
        if "--color" in flags:
            args.extend(["--color", "never"])
        if "--sandbox" in flags:
            args.extend(["--sandbox", "read-only"])
        if (
            "--skip-git-repo-check" in flags
            and not self._args_include_flag(extra_args, "--skip-git-repo-check")
        ):
            args.append("--skip-git-repo-check")
        if (
            ("-c" in flags or "--config" in flags)
            and not self._args_configure_reasoning(extra_args)
        ):
            config_flag = "-c" if "-c" in flags else "--config"
            args.extend([config_flag, self._DEFAULT_REASONING_CONFIG])
        if self.use_json and "--json" in flags:
            args.append("--json")
        args.extend(extra_args)
        if output_path and "--output-last-message" in flags:
            args.extend(["--output-last-message", output_path])
        args.append("-")
        return args

    def _run_codex_exec(self, prompt: str) -> str:
        flags = self._get_supported_flags()
        output_path: Optional[str] = None
        temp_dir: Optional[_LocalTempDir] = None

        if self.use_output_last_message and "--output-last-message" in flags:
            temp_dir = _LocalTempDir(prefix="horizon-codex-")
            output_path = str(Path(temp_dir.name) / "last-message.txt")
        elif "--skip-git-repo-check" in flags:
            temp_dir = _LocalTempDir(prefix="horizon-codex-")

        args = self._build_args(output_path=output_path, flags=flags)
        cwd = temp_dir.name if temp_dir is not None and "--skip-git-repo-check" in flags else None

        try:
            try:
                result = subprocess.run(
                    args,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_sec,
                    check=False,
                    cwd=cwd,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"Codex CLI timed out after {self.timeout_sec} seconds."
                ) from exc
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Codex CLI command not found: {self.command!r}. Install Codex CLI and run `codex login`."
                ) from exc

            artifact_text = self._read_artifact(output_path)
            stdout = result.stdout or ""
            stderr = result.stderr or ""

            if result.returncode != 0:
                raise RuntimeError(
                    "Codex CLI failed with exit code "
                    f"{result.returncode}. stderr: {stderr.strip()[:2000]} "
                    f"stdout: {stdout.strip()[:2000]}"
                )

            if artifact_text:
                return artifact_text

            if self.use_json and "--json" in flags:
                parsed = self._parse_jsonl_stdout(stdout)
                if parsed:
                    return parsed

            return stdout.strip()
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    @staticmethod
    def _read_artifact(output_path: Optional[str]) -> str:
        if not output_path:
            return ""
        path = Path(output_path)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace").strip()

    @staticmethod
    def _parse_jsonl_stdout(stdout: str) -> str:
        final_text = ""
        errors = []

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = str(event.get("type") or event.get("event") or "").lower()
            msg = event.get("msg")
            if event_type == "error":
                errors.append(str(event.get("message") or event))
            if isinstance(msg, dict):
                msg_type = str(msg.get("type") or "").lower()
                content = msg.get("content")
                if msg_type == "error":
                    errors.append(str(content or msg))
                elif (
                    msg_type in {"text", "assistant_message", "agent_message"}
                    and isinstance(content, str)
                ):
                    final_text = content
            elif event_type in {"agentmessage", "assistant_message", "message"}:
                content = event.get("content") or event.get("message")
                if isinstance(content, str):
                    final_text = content

        if errors:
            raise RuntimeError(
                f"Codex CLI reported an error event: {'; '.join(errors)[:2000]}"
            )
        return final_text.strip()


class AnthropicClient(AIClient):
    """Client for Anthropic Claude models."""

    def __init__(self, config: AIConfig):
        """Initialize Anthropic client.

        Args:
            config: AI configuration
        """
        self.config = config

        if not config.api_key_env:
            raise ValueError("Missing API key environment variable name for Anthropic provider")

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key: {config.api_key_env}")

        kwargs = {"api_key": api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url

        self.client = AsyncAnthropic(**kwargs)
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion using Claude.

        Args:
            system: System prompt
            user: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        message = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        usage = getattr(message, "usage", None)
        if usage is not None:
            record_usage(
                "anthropic",
                input_tokens=getattr(usage, "input_tokens", 0),
                output_tokens=getattr(usage, "output_tokens", 0),
            )
        return message.content[0].text


class OpenAIClient(AIClient):
    """Client for OpenAI models."""

    def __init__(self, config: AIConfig):
        """Initialize OpenAI client.

        Args:
            config: AI configuration
        """
        self.config = config

        if not config.api_key_env:
            raise ValueError("Missing API key environment variable name for OpenAI provider")

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key: {config.api_key_env}")

        kwargs = {"api_key": api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url

        self.client = AsyncOpenAI(**kwargs)
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion using OpenAI.

        Args:
            system: System prompt
            user: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            record_usage(
                "openai",
                input_tokens=getattr(usage, "prompt_tokens", 0),
                output_tokens=getattr(usage, "completion_tokens", 0),
            )
        return response.choices[0].message.content


class MiniMaxClient(AIClient):
    """Client for MiniMax models via OpenAI-compatible API."""

    def __init__(self, config: AIConfig):
        """Initialize MiniMax client.

        Args:
            config: AI configuration
        """
        self.config = config

        if not config.api_key_env:
            raise ValueError("Missing API key environment variable name for MiniMax provider")

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key: {config.api_key_env}")

        kwargs = {
            "api_key": api_key,
            "base_url": config.base_url or "https://api.minimax.io/v1",
        }

        self.client = AsyncOpenAI(**kwargs)
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion using MiniMax.

        MiniMax requires temperature in (0.0, 1.0] and does not support
        response_format, so we rely on prompt engineering for JSON output.

        Args:
            system: System prompt
            user: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        # MiniMax temperature must be in (0.0, 1.0]; clamp 0 to a small value
        if temperature <= 0:
            temperature = 0.01

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            record_usage(
                "minimax",
                input_tokens=getattr(usage, "prompt_tokens", 0),
                output_tokens=getattr(usage, "completion_tokens", 0),
            )
        return response.choices[0].message.content


class AliClient(AIClient):
    """Client for Alibaba DashScope (OpenAI-compatible API)."""

    def __init__(self, config: AIConfig):
        """Initialize DashScope client.

        Args:
            config: AI configuration
        """
        self.config = config

        if not config.api_key_env:
            raise ValueError("Missing API key environment variable name for Ali provider")

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key: {config.api_key_env}")

        kwargs = {
            "api_key": api_key,
            "base_url": config.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
        self.client = AsyncOpenAI(**kwargs)
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion using DashScope.

        Args:
            system: System prompt
            user: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content


class GeminiClient(AIClient):
    """Client for Google Gemini models."""

    def __init__(self, config: AIConfig):
        """Initialize Gemini client.

        Args:
            config: AI configuration
        """
        self.config = config

        if not config.api_key_env:
            raise ValueError("Missing API key environment variable name for Gemini provider")

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key: {config.api_key_env}")

        self.client = genai.Client(api_key=api_key)
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion using Gemini.

        Args:
            system: System prompt
            user: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json"
            )
        )
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            total = getattr(usage, "total_token_count", 0) or 0
            prompt = getattr(usage, "prompt_token_count", 0) or 0
            completion = max(0, total - prompt)
            record_usage("gemini", input_tokens=prompt, output_tokens=completion)
        return response.text


def create_ai_client(config: AIConfig) -> AIClient:
    """Factory function to create appropriate AI client.

    Args:
        config: AI configuration

    Returns:
        AIClient: Initialized AI client

    Raises:
        ValueError: If provider is not supported
    """
    if config.provider == AIProvider.ANTHROPIC:
        return AnthropicClient(config)
    elif config.provider == AIProvider.CODEX_CLI:
        return CodexCliClient(config)
    elif config.provider == AIProvider.OPENAI:
        return OpenAIClient(config)
    elif config.provider == AIProvider.ALI:
        return AliClient(config)
    elif config.provider == AIProvider.GEMINI:
        return GeminiClient(config)
    elif config.provider == AIProvider.DOUBAO:
        return OpenAIClient(config)
    elif config.provider == AIProvider.MINIMAX:
        return MiniMaxClient(config)
    else:
        raise ValueError(f"Unsupported AI provider: {config.provider}")
