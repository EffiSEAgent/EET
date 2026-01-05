# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""OpenAI API client wrapper with tool integration using litellm."""

import json
import logging
from typing import Any, override

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from trae_agent.tools.base import Tool, ToolCall, ToolResult
from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.base_client import BaseLLMClient
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse, LLMUsage

logger = logging.getLogger("openai_client")


class OpenAIClient(BaseLLMClient):
    """OpenAI client wrapper with tool schema generation using litellm."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.message_history: list[dict[str, Any]] = []

    @override
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    def _create_litellm_response(
        self,
        messages: list[dict[str, Any]],
        model_config: ModelConfig,
        tools: list[dict[str, Any]] | None,
    ):
        """Create a response using litellm. This method will be decorated with retry logic."""
        kwargs: dict[str, Any] = {
            "model": model_config.model,
            "messages": messages,
        }
        
        # Set API key and base URL for this request
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url
        
        # Add temperature if not o3/o4-mini/gpt-5 models
        if (
            "o3" not in model_config.model
            and "o4-mini" not in model_config.model
            and "gpt-5" not in model_config.model
        ):
            kwargs["temperature"] = model_config.temperature
        
        # kwargs["top_p"] = model_config.top_p
        if model_config.max_tokens:
            kwargs["max_tokens"] = model_config.max_tokens
        
        if tools:
            kwargs["tools"] = tools
        
        # Enable caching to reduce token costs for repeated requests
        # Using ephemeral cache which is automatically managed by OpenAI
        # Note: cache_control parameter temporarily disabled due to compatibility issues
        # Some API providers (via litellm) don't support this parameter
        # kwargs["cache_control"] = {"type": "ephemeral"}
        
        try:
            return litellm.completion(**kwargs)
        except litellm.exceptions.AuthenticationError as e:
            if hasattr(e, "message"):
                e.message += " Please check your API key configuration."
            raise e

    @override
    def chat(
        self,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Send chat messages to OpenAI with optional tool support using litellm."""
        parsed_messages = self.parse_messages(messages)

        # Update message history with new messages (including tool_results)
        if reuse_history:
            self.message_history.extend(parsed_messages)
        else:
            self.message_history = parsed_messages.copy()

        tool_schemas = None
        if tools:
            tool_schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.get_input_schema(),
                    },
                }
                for tool in tools
            ]

        # Use the updated message history for API call
        api_call_messages = self.message_history.copy()

        # Apply retry logic using tenacity
        # BadRequestError should not be retried as it indicates client-side errors
        non_retryable_exceptions = (
            litellm.exceptions.UnsupportedParamsError,
            litellm.exceptions.NotFoundError,
            litellm.exceptions.PermissionDeniedError,
            litellm.exceptions.ContextWindowExceededError,
            litellm.exceptions.APIError,
            litellm.exceptions.AuthenticationError,
            KeyboardInterrupt,
        )
        # Add BadRequestError if it exists in litellm
        try:
            if hasattr(litellm.exceptions, "BadRequestError"):
                non_retryable_exceptions = non_retryable_exceptions + (
                    litellm.exceptions.BadRequestError,
                )
        except AttributeError:
            pass
        
        @retry(
            stop=stop_after_attempt(model_config.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            retry=retry_if_not_exception_type(non_retryable_exceptions),
        )
        def _call_with_retry():
            return self._create_litellm_response(api_call_messages, model_config, tool_schemas)
        
        response = _call_with_retry()

        # Extract response content and tool calls
        choice = response.choices[0]
        message = choice.message

        content = message.content or ""
        tool_calls: list[ToolCall] = []
        
        # Handle tool calls if present
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_call_list = []
            for tool_call in message.tool_calls:
                if hasattr(tool_call, "function"):
                    function = tool_call.function
                    call_id = getattr(tool_call, "id", None) or getattr(tool_call, "call_id", None)
                    tool_calls.append(
                        ToolCall(
                            call_id=call_id or "",
                            name=function.name,
                            arguments=json.loads(function.arguments) if isinstance(function.arguments, str) else (function.arguments or {}),
                            id=call_id or "",
                        )
                    )
                    tool_call_list.append({
                        "id": call_id or "",
                        "type": "function",
                        "function": {
                            "name": function.name,
                            "arguments": json.dumps(function.arguments) if not isinstance(function.arguments, str) else function.arguments,
                        },
                    })
            
            # Add assistant message with tool calls to history
            self.message_history.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_call_list,
            })
        elif content:
            # Add assistant message to history if there's content but no tool calls
            self.message_history.append({
                "role": "assistant",
                "content": content,
            })

        # Extract usage information
        usage = None
        if hasattr(response, "usage") and response.usage:
            # Extract basic token counts
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
            
            # Extract cache token information from prompt_tokens_details
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 0
            
            # Check for prompt_tokens_details (OpenAI API format)
            if hasattr(response.usage, "prompt_tokens_details"):
                prompt_details = response.usage.prompt_tokens_details
                if prompt_details:
                    # Extract cached_tokens (cache_read)
                    if hasattr(prompt_details, "cached_tokens"):
                        cache_read_input_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
                    # Also check for cache_creation_tokens if available
                    if hasattr(prompt_details, "cache_creation_tokens"):
                        cache_creation_input_tokens = getattr(prompt_details, "cache_creation_tokens", 0) or 0
            
            # Fallback: check if usage has direct cache fields (some API formats)
            if cache_read_input_tokens == 0:
                cache_read_input_tokens = getattr(response.usage, "cache_read_tokens", 0) or 0
                if cache_read_input_tokens == 0:
                    cache_read_input_tokens = getattr(response.usage, "cached_tokens", 0) or 0
            
            # Extract reasoning tokens from completion_tokens_details if available
            reasoning_tokens = 0
            if hasattr(response.usage, "completion_tokens_details"):
                completion_details = response.usage.completion_tokens_details
                if completion_details and hasattr(completion_details, "reasoning_tokens"):
                    reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0
            
            usage = LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                reasoning_tokens=reasoning_tokens,
            )

        finish_reason = getattr(choice, "finish_reason", None) or "stop"

        llm_response = LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            finish_reason=finish_reason,
            tool_calls=tool_calls if len(tool_calls) > 0 else None,
        )

        # Record trajectory if recorder is available
        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_interaction(
                messages=messages,
                response=llm_response,
                provider="openai",
                model=model_config.model,
                tools=tools,
            )

        return llm_response

    def parse_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Parse the messages to standard chat format."""
        chat_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.tool_result:
                # Tool result is added as a tool message
                chat_messages.append(self.parse_tool_call_result(msg.tool_result))
            elif msg.tool_call:
                # Tool call is part of assistant message
                chat_messages.append(self.parse_tool_call(msg.tool_call))
            else:
                if msg.role == "system":
                    if not msg.content:
                        raise ValueError("System message content is required")
                    chat_messages.append({"role": "system", "content": msg.content})
                elif msg.role == "user":
                    if not msg.content:
                        raise ValueError("User message content is required")
                    chat_messages.append({"role": "user", "content": msg.content})
                elif msg.role == "assistant":
                    # Assistant messages can have None content (OpenAI API allows this)
                    chat_messages.append({"role": "assistant", "content": msg.content})
                else:
                    raise ValueError(f"Invalid message role: {msg.role}")
        return chat_messages

    def parse_tool_call(self, tool_call: ToolCall) -> dict[str, Any]:
        """Parse the tool call from the LLM response."""
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call.id or tool_call.call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments),
                    },
                }
            ],
        }

    def parse_tool_call_result(self, tool_call_result: ToolResult) -> dict[str, Any]:
        """Parse the tool call result from the LLM response to standard format."""
        result_content: str = ""
        if tool_call_result.result is not None:
            result_content += str(tool_call_result.result)
        if tool_call_result.error:
            result_content += f"\nError: {tool_call_result.error}"
        result_content = result_content.strip()

        return {
            "role": "tool",
            "content": result_content,
            "tool_call_id": tool_call_result.call_id,
        }
