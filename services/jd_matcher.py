from __future__ import annotations

from models.ats_report import JobDescriptionAnalysis
from services.ats_agent import AzureJSONClient, AzureOpenAISettings
from utils.prompts import JOB_ANALYSIS_PROMPT, SYSTEM_PROMPT


class JDMatcher:
    def __init__(self, settings: AzureOpenAISettings | None = None) -> None:
        self.json_client = AzureJSONClient(settings)

    def analyze(self, job_description: str) -> JobDescriptionAnalysis:
        if not job_description.strip():
            raise ValueError("Job description is required for ATS analysis.")
        return self.json_client.generate_json(
            system_prompt=SYSTEM_PROMPT,
            task_prompt=JOB_ANALYSIS_PROMPT,
            payload={"job_description": job_description},
            response_model=JobDescriptionAnalysis,
        )
