from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, TypeVar

from openai import AzureOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from models.ats_report import (
    ATSReport,
    ATSScoreBreakdown,
    CareerRecommendation,
    JobDescriptionAnalysis,
    ResumeData,
    SkillAnalysis,
)
from utils.prompts import (
    ATS_MATCHING_PROMPT,
    CAREER_RECOMMENDATION_PROMPT,
    RESUME_EXTRACTION_PROMPT,
    SKILL_ANALYSIS_PROMPT,
    SYSTEM_PROMPT,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(slots=True)
class AzureOpenAISettings:
    endpoint: str = ""
    api_key: str = ""
    deployment_name: str = ""
    api_version: str = "2025-01-01-preview"

    @classmethod
    def from_env(cls) -> "AzureOpenAISettings":
        return cls(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint.strip() and self.api_key.strip() and self.deployment_name.strip())


class AzureJSONClient:
    def __init__(self, settings: AzureOpenAISettings | None = None) -> None:
        self.settings = settings or AzureOpenAISettings.from_env()
        self._client: AzureOpenAI | None = None

    @property
    def client(self) -> AzureOpenAI:
        if not self.settings.is_configured:
            raise RuntimeError("Azure OpenAI endpoint, API key, and deployment name are required.")
        if self._client is None:
            self._client = AzureOpenAI(
                azure_endpoint=self.settings.endpoint,
                api_key=self.settings.api_key,
                api_version=self.settings.api_version,
            )
        return self._client

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
    def generate_json(
        self,
        *,
        system_prompt: str,
        task_prompt: str,
        payload: dict[str, Any],
        response_model: type[ModelT],
        temperature: float = 0.1,
    ) -> ModelT:
        response = self.client.chat.completions.create(
            model=self.settings.deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"{task_prompt}\n\nInput JSON:\n{json.dumps(payload, ensure_ascii=False)}",
                },
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        data = self._parse_json(content)
        return response_model.model_validate(data)

    def _parse_json(self, content: str) -> dict[str, Any]:
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
            stripped = re.sub(r"```$", "", stripped).strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))


class ATSAgent:
    def __init__(self, settings: AzureOpenAISettings | None = None) -> None:
        self.json_client = AzureJSONClient(settings)

    def extract_resume_data(self, resume_text: str, source_file: str = "") -> ResumeData:
        resume = self.json_client.generate_json(
            system_prompt=SYSTEM_PROMPT,
            task_prompt=RESUME_EXTRACTION_PROMPT,
            payload={"resume_text": resume_text[:120_000]},
            response_model=ResumeData,
        )
        resume.raw_text = resume_text
        resume.source_file = source_file
        return resume

    def analyze_skills(self, resume: ResumeData) -> SkillAnalysis:
        analysis = self.json_client.generate_json(
            system_prompt=SYSTEM_PROMPT,
            task_prompt=SKILL_ANALYSIS_PROMPT,
            payload={"resume": resume.model_dump(exclude={"raw_text"})},
            response_model=SkillAnalysis,
        )
        analysis.all_skills = self._dedupe([*analysis.all_skills, *resume.skills])
        if not analysis.skill_distribution:
            analysis.skill_distribution = self._skill_distribution(analysis)
        return analysis

    def generate_ats_report(
        self,
        resume: ResumeData,
        skill_analysis: SkillAnalysis,
        job_analysis: JobDescriptionAnalysis,
        job_description: str,
    ) -> ATSReport:
        report = self.json_client.generate_json(
            system_prompt=SYSTEM_PROMPT,
            task_prompt=ATS_MATCHING_PROMPT,
            payload={
                "resume": resume.model_dump(exclude={"raw_text"}),
                "skill_analysis": skill_analysis.model_dump(),
                "job_analysis": job_analysis.model_dump(),
                "job_description": job_description,
            },
            response_model=ATSReport,
        )
        return self._apply_scoring_policy(report, resume, skill_analysis, job_analysis)

    def generate_career_recommendations(
        self,
        resume: ResumeData,
        skill_analysis: SkillAnalysis,
        job_analysis: JobDescriptionAnalysis | None = None,
        ats_report: ATSReport | None = None,
    ) -> CareerRecommendation:
        return self.json_client.generate_json(
            system_prompt=SYSTEM_PROMPT,
            task_prompt=CAREER_RECOMMENDATION_PROMPT,
            payload={
                "resume": resume.model_dump(exclude={"raw_text"}),
                "skill_analysis": skill_analysis.model_dump(),
                "job_analysis": job_analysis.model_dump() if job_analysis else {},
                "ats_report": ats_report.model_dump() if ats_report else {},
            },
            response_model=CareerRecommendation,
        )

    def _apply_scoring_policy(
        self,
        report: ATSReport,
        resume: ResumeData,
        skill_analysis: SkillAnalysis,
        job_analysis: JobDescriptionAnalysis,
    ) -> ATSReport:
        resume_skills = self._dedupe(
            [
                *resume.skills,
                *skill_analysis.all_skills,
                *skill_analysis.technical_skills,
                *skill_analysis.cloud_skills,
                *skill_analysis.devops_skills,
                *skill_analysis.genai_skills,
                *skill_analysis.programming_languages,
                *skill_analysis.tools,
            ]
        )
        required = self._dedupe(job_analysis.required_skills)
        preferred = self._dedupe([*job_analysis.preferred_skills, *job_analysis.soft_skills])

        resume_map = self._normalized_map(resume_skills)
        required_map = self._normalized_map(required)
        preferred_map = self._normalized_map(preferred)
        resume_norm = set(resume_map)

        matched_required = sorted(set(required_map) & resume_norm)
        matched_preferred = sorted(set(preferred_map) & resume_norm)
        missing_required = sorted(set(required_map) - resume_norm)
        missing_preferred = sorted(set(preferred_map) - resume_norm)

        if required and preferred:
            required_ratio = len(matched_required) / max(1, len(required))
            preferred_ratio = len(matched_preferred) / max(1, len(preferred))
            skills_score = round(40 * ((required_ratio * 0.75) + (preferred_ratio * 0.25)))
        elif required:
            skills_score = round(40 * len(matched_required) / max(1, len(required)))
        elif preferred:
            skills_score = round(40 * len(matched_preferred) / max(1, len(preferred)))
        else:
            skills_score = 40 if resume_skills else 0

        experience_score = self._experience_score(resume.total_years_experience, job_analysis.years_experience)
        education_score = self._education_score(resume, job_analysis)
        certification_score = self._certification_score(resume.certifications, job_analysis.certifications)

        report.score_breakdown = ATSScoreBreakdown(
            skills=skills_score,
            experience=experience_score,
            education=education_score,
            certifications=certification_score,
        )
        report.ats_score = report.score_breakdown.total
        report.matched_skills = self._dedupe(
            [required_map[item] for item in matched_required] + [preferred_map[item] for item in matched_preferred]
        )
        report.missing_skills = self._dedupe(
            [required_map[item] for item in missing_required] + [preferred_map[item] for item in missing_preferred]
        )
        report.strong_skills = self._dedupe([*report.strong_skills, *report.matched_skills])[:12]
        report.recommended_skills = self._dedupe([*report.recommended_skills, *report.missing_skills])[:15]
        if report.missing_skills and not report.recommendations:
            report.recommendations = [
                f"Add measurable project or work evidence for {skill}."
                for skill in report.missing_skills[:5]
            ]
        return report

    def _experience_score(self, resume_years: float, required_years: float) -> int:
        if required_years <= 0:
            return 30 if resume_years > 0 else 18
        return round(30 * min(1.0, resume_years / required_years))

    def _education_score(self, resume: ResumeData, job_analysis: JobDescriptionAnalysis) -> int:
        if not resume.education:
            return 0
        if not job_analysis.education_requirements:
            return 20
        resume_text = " ".join(
            f"{item.degree} {item.field_of_study} {item.institution}" for item in resume.education
        ).lower()
        requirement_text = " ".join(job_analysis.education_requirements).lower()
        degree_terms = ["bachelor", "master", "phd", "doctorate", "computer", "engineering", "science"]
        if any(term in resume_text and term in requirement_text for term in degree_terms):
            return 20
        return 12

    def _certification_score(self, resume_certifications: list[str], required_certifications: list[str]) -> int:
        if not required_certifications:
            return 10 if resume_certifications else 6
        resume_map = self._normalized_map(resume_certifications)
        required_map = self._normalized_map(required_certifications)
        matched = set(resume_map) & set(required_map)
        return round(10 * len(matched) / max(1, len(required_map)))

    def _skill_distribution(self, analysis: SkillAnalysis) -> dict[str, int]:
        return {
            "Technical": len(analysis.technical_skills),
            "Cloud": len(analysis.cloud_skills),
            "DevOps": len(analysis.devops_skills),
            "GenAI": len(analysis.genai_skills),
            "Soft Skills": len(analysis.soft_skills),
            "Tools": len(analysis.tools),
            "Programming": len(analysis.programming_languages),
            "Other": len(analysis.other_skills),
        }

    def _normalized_map(self, values: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for value in values:
            normalized = self._normalize(value)
            if normalized and normalized not in result:
                result[normalized] = value
        return result

    def _normalize(self, value: str) -> str:
        normalized = value.lower().strip()
        replacements = {
            "nodejs": "node.js",
            "node js": "node.js",
            "gen ai": "genai",
            "generative ai": "genai",
            "k8s": "kubernetes",
        }
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)
        return re.sub(r"[^a-z0-9+#.]+", "", normalized)

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            normalized = self._normalize(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(value.strip())
        return result
