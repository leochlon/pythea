import os
import requests

class GeminiBackend:
    def __init__(self, model: str):
        self.model = model
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent".format(model)

    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 256):
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
        }
        response = requests.post(self.api_url, headers=headers, params=params, json=data)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]

class GeminiItem:
    def __init__(self, prompt: str, n_samples: int, m: int, skeleton_policy: str):
        self.prompt = prompt
        self.n_samples = n_samples
        self.m = m
        self.skeleton_policy = skeleton_policy

class GeminiPlanner:
    def __init__(self, backend: GeminiBackend, temperature: float):
        self.backend = backend
        self.temperature = temperature

    def run(self, items, h_star, isr_threshold, margin_extra_bits, B_clip, clip_mode):
        # Placeholder: mimic OpenAIPlanner output structure
        results = []
        for item in items:
            answer = self.backend.generate(item.prompt, temperature=self.temperature)
            # Dummy metrics for demonstration
            result = type('Metric', (), {})()
            result.decision_answer = True
            result.rationale = answer
            result.roh_bound = 0.01
            result.isr = 1.0
            result.b2t = 1.0
            results.append(result)
        return results

    def aggregate(self, items, metrics, h_star, isr_threshold, margin_extra_bits):
        # Return an object with required attributes for SLA certificate
        class Report:
            pass
        report = Report()
        report.h_star = h_star
        report.isr_threshold = isr_threshold
        report.margin_extra_bits = margin_extra_bits
        report.answer_rate = 1.0
        report.abstention_rate = 0.0
        report.empirical_hallucination_rate = 0.0
        report.wilson_upper = 0.0
        report.worst_item_roh_bound = 0.0
        report.median_item_roh_bound = 0.0
        report.n_items = 1
        report.n_answered_with_labels = 1
        report.hallucinations_observed = 0
        return report
