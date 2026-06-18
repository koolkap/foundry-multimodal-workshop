from __future__ import annotations

import re

from models.ats_report import JobDescriptionAnalysis
from services.ats_agent import CERTIFICATION_TERMS, SKILL_CATALOG


class JDMatcher:
    def analyze(self, job_description: str) -> JobDescriptionAnalysis:
        if not job_description.strip():
            raise ValueError("Job description is required for ATS analysis.")

        required_text = self._section_text(job_description, ["required", "requirements", "must have", "minimum"])
        preferred_text = self._section_text(job_description, ["preferred", "nice to have", "bonus", "plus"])
        required_skills = self._detect_skills(required_text or job_description)
        preferred_skills = [skill for skill in self._detect_skills(preferred_text) if skill not in required_skills]
        soft_skills = [skill for skill in SKILL_CATALOG["soft_skills"] if self._contains_term(job_description, skill)]

        return JobDescriptionAnalysis(
            role_title=self._guess_role_title(job_description),
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            soft_skills=soft_skills,
            years_experience=self._years_experience(job_description),
            certifications=[cert for cert in CERTIFICATION_TERMS if self._contains_term(job_description, cert)],
            education_requirements=self._education_requirements(job_description),
            responsibilities=self._responsibilities(job_description),
            seniority_level=self._seniority(job_description),
            domain=self._domain(job_description),
        )

    def _detect_skills(self, text: str) -> list[str]:
        detected: list[str] = []
        for skills in SKILL_CATALOG.values():
            for skill in skills:
                if self._contains_term(text, skill):
                    detected.append(skill)
        return self._dedupe(detected)

    def _section_text(self, text: str, section_markers: list[str]) -> str:
        lines = text.splitlines()
        capture = False
        captured: list[str] = []
        for line in lines:
            lowered = line.lower()
            if any(marker in lowered for marker in section_markers):
                capture = True
                captured.append(line)
                continue
            if capture and re.match(r"^\s*[A-Z][A-Za-z ]{2,30}:?\s*$", line):
                break
            if capture:
                captured.append(line)
        return "\n".join(captured)

    def _years_experience(self, text: str) -> float:
        patterns = [
            r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s+(?:of\s+)?experience",
            r"experience\s*(?:of\s*)?(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)",
            r"(\d+(?:\.\d+)?)\s*-\s*\d+\s*(?:years|yrs)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 0.0

    def _education_requirements(self, text: str) -> list[str]:
        requirements: list[str] = []
        for term in ["Bachelor", "Master", "PhD", "Computer Science", "Engineering", "Degree"]:
            if self._contains_term(text, term):
                requirements.append(term)
        return self._dedupe(requirements)

    def _responsibilities(self, text: str) -> list[str]:
        items: list[str] = []
        for line in text.splitlines():
            cleaned = line.strip(" -\t")
            if 20 <= len(cleaned) <= 180:
                lowered = cleaned.lower()
                if any(word in lowered for word in ["build", "design", "develop", "manage", "lead", "deploy"]):
                    items.append(cleaned)
        return items[:10]

    def _guess_role_title(self, text: str) -> str:
        for line in text.splitlines()[:8]:
            cleaned = line.strip(" -\t")
            if 4 <= len(cleaned) <= 80 and not cleaned.endswith("."):
                return cleaned
        return "Target Role"

    def _seniority(self, text: str) -> str:
        lowered = text.lower()
        if any(term in lowered for term in ["principal", "staff", "lead"]):
            return "Lead"
        if "senior" in lowered or "sr." in lowered:
            return "Senior"
        if "junior" in lowered or "entry" in lowered:
            return "Junior"
        return "Mid"

    def _domain(self, text: str) -> str:
        lowered = text.lower()
        if any(term in lowered for term in ["ai", "ml", "machine learning", "llm", "genai"]):
            return "AI/ML"
        if "cloud" in lowered or "azure" in lowered or "aws" in lowered:
            return "Cloud"
        if "frontend" in lowered or "react" in lowered:
            return "Frontend"
        return "Software"

    def _contains_term(self, text: str, term: str) -> bool:
        pattern = r"(?<![a-zA-Z0-9+#.])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![a-zA-Z0-9+#.])"
        return bool(re.search(pattern, text, flags=re.IGNORECASE))

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = re.sub(r"[^a-z0-9+#.]+", "", value.lower())
            if key and key not in seen:
                seen.add(key)
                result.append(value)
        return result
