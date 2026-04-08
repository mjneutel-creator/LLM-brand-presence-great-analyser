import os
import json
import requests

class LLMError(RuntimeError):
    pass

class BaseConnector:
    name = "base"
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int = 60):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OfflineMockConnector(BaseConnector):
    name = "offline_mock"

    def __init__(self, model: str = "mock", **kwargs):
        super().__init__(api_key="mock", model=model, timeout=10)

    def generate(self, prompt: str) -> str:
        # Deterministic-ish mock so you can use the tool without keys.
        seed = sum(ord(c) for c in prompt) % 4
        variants = [
            "is often associated with stability, scale, and mainstream trust. It is frequently referenced for retail reach and pragmatic initiatives, but it can be framed as less innovative than smaller challengers.",
            "is commonly discussed in relation to ESG commitments and financing. Some narratives highlight progress, while criticism may focus on pace of change and perceived inconsistencies.",
            "tends to appear in comparisons about customer experience, product breadth, and reliability. Risks discussed can include legacy systems, service issues, and scrutiny on sustainability claims.",
            "is described through a mix of strengths (reach, resilience) and critiques (bureaucracy, conservatism). Comparative framing often depends on topic emphasis (innovation vs responsibility)."
        ]
        return f"Mock response: {variants[seed]}"


class OpenAIConnector(BaseConnector):
    name = "openai"

    def generate(self, prompt: str) -> str:
        # Preferred: official SDK if installed; fallback to HTTPS.
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model or os.getenv("OPENAI_MODEL", "gpt-4.1"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content
        except Exception:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model or os.getenv("OPENAI_MODEL", "gpt-4.1"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
            if r.status_code >= 400:
                raise LLMError(f"OpenAI API error {r.status_code}: {r.text}")
            return r.json()["choices"][0]["message"]["content"]


class AnthropicConnector(BaseConnector):
    name = "anthropic"

    def generate(self, prompt: str) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            msg = client.messages.create(
                model=self.model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                max_tokens=900,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": self.model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                "max_tokens": 900,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
            if r.status_code >= 400:
                raise LLMError(f"Anthropic API error {r.status_code}: {r.text}")
            j = r.json()
            return "".join(block.get("text", "") for block in j.get("content", []))


class GeminiConnector(BaseConnector):
    name = "gemini"

    def generate(self, prompt: str) -> str:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model_name = self.model or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            return getattr(resp, "text", str(resp))
        except Exception:
            model_name = self.model or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0}}
            r = requests.post(url, headers={"content-type": "application/json"}, data=json.dumps(payload), timeout=self.timeout)
            if r.status_code >= 400:
                raise LLMError(f"Gemini API error {r.status_code}: {r.text}")
            j = r.json()
            try:
                return j["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                return json.dumps(j)


class MistralConnector(BaseConnector):
    name = "mistral"

    def generate(self, prompt: str) -> str:
        try:
            from mistralai.client import MistralClient
            from mistralai.models.chat_completion import ChatMessage
            client = MistralClient(api_key=self.api_key)
            resp = client.chat(
                model=self.model or os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0,
            )
            return resp.choices[0].message.content
        except Exception:
            url = "https://api.mistral.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model or os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
            if r.status_code >= 400:
                raise LLMError(f"Mistral API error {r.status_code}: {r.text}")
            return r.json()["choices"][0]["message"]["content"]


def build_connectors(keys: dict, models: dict, include_offline: bool = True):
    connectors = {}
    if include_offline:
        connectors["Offline (no keys)"] = OfflineMockConnector()

    if keys.get("OPENAI_API_KEY"):
        connectors["OpenAI"] = OpenAIConnector(keys["OPENAI_API_KEY"], model=models.get("OPENAI_MODEL"))

    if keys.get("ANTHROPIC_API_KEY"):
        connectors["Claude"] = AnthropicConnector(keys["ANTHROPIC_API_KEY"], model=models.get("ANTHROPIC_MODEL"))

    if keys.get("GOOGLE_API_KEY"):
        connectors["Gemini"] = GeminiConnector(keys["GOOGLE_API_KEY"], model=models.get("GEMINI_MODEL"))

    if keys.get("MISTRAL_API_KEY"):
        connectors["Mistral"] = MistralConnector(keys["MISTRAL_API_KEY"], model=models.get("MISTRAL_MODEL"))

    return connectors
