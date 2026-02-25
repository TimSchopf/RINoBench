import time
import re
import json
import asyncio
from functools import partial
from openai import OpenAI
import instructor
from deepeval.models import DeepEvalBaseLLM

class LLMClient:
    """
    Unified client manager for OpenAI LLMs with and without Instructor schema support.

    This class initializes both:
      - a standard OpenAI client (for normal text generation)
      - an Instructor-wrapped OpenAI client (for schema-based structured responses)
    """

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        """
        Initialize both OpenAI and Instructor clients.

        Args:
            api_key (str): OpenAI (or OpenAI-compatible) API key.
            base_url (str, optional): Base API endpoint. Defaults to OpenAI's URL.
        """
        # Base OpenAI client
        self.base_client = OpenAI(base_url=base_url, api_key=api_key)

        # Instructor client (wraps the OpenAI client for structured outputs)
        self.instructor_client = instructor.from_openai(OpenAI(base_url=base_url, api_key=api_key), mode=instructor.Mode.JSON)

    def get_client(self, use_schema: bool = False):
        """Return the appropriate client depending on schema usage."""
        return self.instructor_client if use_schema else self.base_client

    def get_base_client(self):
        """Return the standard OpenAI client (no schema validation)."""
        return self.base_client

    def get_instructor_client(self):
        """Return the Instructor-wrapped client for schema-based structured responses."""
        return self.instructor_client

def extract_json_from_text(text: str):
    """
    Attempts to find the first valid JSON object or array inside a text string
    and parse it. Returns the parsed object.
    Raises ValueError if no valid JSON is found.
    """
    # Regular expression to match JSON objects or arrays
    json_pattern = r"(\{.*?\}|\[.*?\])"

    matches = re.findall(json_pattern, text, flags=re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue  # try next match

    # If none succeeded
    raise ValueError("No valid JSON could be parsed from the text")

def call_llm_with_retries(client: LLMClient, llm_name: str, user_prompt: str, schema=None, system_prompt: str | None = None, temperature: float | None = 0, return_json: bool = False, max_retries: int = 3, retry_backoff: float = 1.0):
    """
    Call an LLM (optionally with schema-based validation and a system prompt) with retry logic.

    Args:
        client: LLM API client instance.
        llm_name (str): Name or ID of the LLM to use.
        user_prompt (str): Input text prompt for the LLM.
        schema (optional): Response schema for structured output validation. If None, schema is not used.
        system_prompt (str, optional): System prompt defining model behavior. If None, no system role is added.
        temperature (float, optional): Sampling temperature. Defaults to 0 (deterministic).
        return_json (bool, optional): If True, checks if the model output contains valid JSON and only returns the JSON. Defaults to False.
        max_retries (int, optional): Maximum number of retries in case of failure. Defaults to 3.
        retry_backoff (float, optional): Initial delay between retries in seconds. Exponential backoff applied. Defaults to 1.0.

    Returns:
        The API response (raw or schema-validated, depending on `schema`).

    Raises:
        RuntimeError: If the call fails after all retry attempts.
    """

    # Choose which client to use (Instructor for schema, plain OpenAI otherwise)
    llm_client = client.get_client(use_schema=schema is not None)

    def call_llm():
        """Helper function to perform one API call (with or without schema/system prompt)."""

        # Build messages dynamically
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Base request payload
        request_args = dict(
            model=llm_name,
            messages=messages,
            temperature=temperature,
        )

        # Add schema validation if provided
        if schema is not None:
            request_args["response_model"] = schema

        # disable fallback LLM if client uses ScaDS.AI API
        if str(llm_client.base_url) == "https://llm.scads.ai/v1/":
            request_args["extra_body"] = {"disable_fallbacks": True}

        # Make the API call
        return llm_client.chat.completions.create(**request_args)

    for attempt in range(max_retries):
        try:
            response = call_llm()

            # Optional: Parse output to JSON
            if return_json:
                try:
                    response.choices[0].message.content = extract_json_from_text(response.choices[0].message.content)
                except (json.JSONDecodeError, AttributeError, KeyError) as e:
                    raise ValueError(f"Invalid JSON output: {str(e)}")

            # ✅ Return the result if successful
            return response

        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = retry_backoff #* (2 ** attempt)
                print(
                    f"⚠️ LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                raise RuntimeError(f"❌ Failed after {max_retries} attempts: {str(e)}")
    return None


# async llm call wrapper
async def call_llm_with_retries_async(*args, **kwargs):
    """
    Wrap synchronous call_llm_with_retries in an executor to make it async.
    Supports both positional and keyword arguments.
    """
    loop = asyncio.get_running_loop()
    func = partial(call_llm_with_retries, *args, **kwargs)
    return await loop.run_in_executor(None, func)


class CustomLLM(DeepEvalBaseLLM):
    """
    Unified LLM wrapper for both ScaDS.AI and OpenAI models.
    Compatible with DeepEval evaluation tools.
    """

    def __init__(self, client, model_name: str, temperature: float | None = 0):
        """
        Initialize the unified LLM wrapper.

        Args:
            client: API client (either ScaDS.AI or OpenAI).
            model_name (str): Model name, e.g. 'gpt-4.1' or 'scads-llm'.
        """
        self.client = client
        self.model_name = model_name
        self.is_scads = self._detect_scads_client()
        self.temperature = temperature

    def _detect_scads_client(self) -> bool:
        """
        Detect whether the client belongs to ScaDS.AI based on attributes or module name.

        Returns:
            bool: True if ScaDS.AI client, False if OpenAI or other compatible client.
        """
        client_module = getattr(self.client.__class__, "__module__", "")
        return "scads" in client_module.lower() or "ScaDS" in str(self.client.__class__)

    def load_model(self):
        """API-based model, so just return the model name."""
        return self.model_name

    def generate(self, prompt: str) -> str:
        """
        Generate text synchronously using ScaDS.AI or OpenAI API.

        Args:
            prompt (str): Input text prompt.
            temperature (float, optional): Sampling temperature. Defaults to 0.

        Returns:
            str: Model-generated text content.
        """
        request_args = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        # Add ScaDS.AI-specific option if applicable
        if self.is_scads:
            request_args["extra_body"] = {"disable_fallbacks": True}

        response = self.client.chat.completions.create(**request_args)
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, temperature: float | None =0) -> str:
        """
        Asynchronous version of generate (compatible with async clients).

        Args:
            prompt (str): Input text prompt.
            temperature (float, optional): Sampling temperature. Defaults to 0.

        Returns:
            str: Model-generated text content.
        """
        request_args = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        if self.is_scads:
            request_args["extra_body"] = {"disable_fallbacks": True}

        # Try async; fall back to sync if client is not async-capable
        try:
            response = await self.client.chat.completions.create(**request_args)
        except TypeError:
            response = self.client.chat.completions.create(**request_args)

        return response.choices[0].message.content

    def get_model_name(self):
        """Return the formatted model name for logging and evaluation."""
        source = "ScaDS.AI" if self.is_scads else "OpenAI"
        return f"{source}-{self.model_name}"