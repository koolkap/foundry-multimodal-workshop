from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    return [str(value).strip()] if str(value).strip() else []


def _clamp_score(value: Any, upper: int) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(upper, score))


def _clamp_probability(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score > 1:
        score = score / 100
    return max(0.0, min(1.0, score))


class ATSBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class ExperienceEntry(ATSBaseModel):
    title: str = ""
    company: str = ""
    location: str = ""
    start_date: str = ""
    end_date: str = ""
    duration: str = ""
    responsibilities: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)

    @field_validator("responsibilities", "technologies", mode="before")
    @classmethod
    def normalize_lists(cls, value: Any) -> list[str]:
        return _coerce_list(value)


class EducationEntry(ATSBaseModel):
    institution: str = ""
    degree: str = ""
    field_of_study: str = ""
    graduation_year: str = ""
    location: str = ""
    achievements: list[str] = Field(default_factory=list)

    @field_validator("achievements", mode="before")
    @classmethod
    def normalize_lists(cls, value: Any) -> list[str]:
        return _coerce_list(value)


class ProjectEntry(ATSBaseModel):
    name: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)

    @field_validator("technologies", "outcomes", mode="before")
    @classmethod
    def normalize_lists(cls, value: Any) -> list[str]:
        return _coerce_list(value)


class ResumeData(ATSBaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    location: str = ""
    professional_summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    total_years_experience: float = 0.0
    raw_text: str = ""
    source_file: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("skills", "certifications", "warnings", mode="before")
    @classmethod
    def normalize_lists(cls, value: Any) -> list[str]:
        return _coerce_list(value)

    @field_validator("total_years_experience", mode="before")
    @classmethod
    def normalize_years(cls, value: Any) -> float:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: Any) -> float:
        return _clamp_probability(value)


class SkillAnalysis(ATSBaseModel):
    technical_skills: list[str] = Field(default_factory=list)
    cloud_skills: list[str] = Field(default_factory=list)
    devops_skills: list[str] = Field(default_factory=list)
    genai_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    programming_languages: list[str] = Field(default_factory=list)
    other_skills: list[str] = Field(default_factory=list)
    all_skills: list[str] = Field(default_factory=list)
    skill_distribution: dict[str, int] = Field(default_factory=dict)

    @field_validator(
        "technical_skills",
        "cloud_skills",
        "devops_skills",
        "genai_skills",
        "soft_skills",
        "tools",
        "programming_languages",
        "other_skills",
        "all_skills",
        mode="before",
    )
    @classmethod
    def normalize_lists(cls, value: Any) -> list[str]:
        return _coerce_list(value)


class JobDescriptionAnalysis(ATSBaseModel):
    role_title: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    years_experience: float = 0.0
    certifications: list[str] = Field(default_factory=list)
    education_requirements: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    seniority_level: str = ""
    domain: str = ""

    @field_validator(
        "required_skills",
        "preferred_skills",
        "soft_skills",
        "certifications",
        "education_requirements",
        "responsibilities",
        mode="before",
    )
    @classmethod
    def normalize_lists(cls, value: Any) -> list[str]:
        return _coerce_list(value)

    @field_validator("years_experience", mode="before")
    @classmethod
    def normalize_years(cls, value: Any) -> float:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0


class ATSScoreBreakdown(ATSBaseModel):
    skills: int = Field(default=0, description="Points out of 40")
    experience: int = Field(default=0, description="Points out of 30")
    education: int = Field(default=0, description="Points out of 20")
    certifications: int = Field(default=0, description="Points out of 10")

    @field_validator("skills", mode="before")
    @classmethod
    def clamp_skills(cls, value: Any) -> int:
        return _clamp_score(value, 40)

    @field_validator("experience", mode="before")
    @classmethod
    def clamp_experience(cls, value: Any) -> int:
        return _clamp_score(value, 30)

    @field_validator("education", mode="before")
    @classmethod
    def clamp_education(cls, value: Any) -> int:
        return _clamp_score(value, 20)

    @field_validator("certifications", mode="before")
    @classmethod
    def clamp_certifications(cls, value: Any) -> int:
        return _clamp_score(value, 10)

    @property
    def total(self) -> int:
        return self.skills + self.experience + self.education + self.certifications


class RoleRecommendation(ATSBaseModel):
    role: str = ""
    fit_score: int = Field(default=0, ge=0, le=100)
    salary_band: str = ""
    rationale: str = ""
    learning_path: list[str] = Field(default_factory=list)
    missing_technologies: list[str] = Field(default_factory=list)

    @field_validator("fit_score", mode="before")
    @classmethod
    def clamp_fit_score(cls, value: Any) -> int:
        return _clamp_score(value, 100)

    @field_validator("learning_path", "missing_technologies", mode="before")
    @classmethod
    def normalize_lists(cls, value: Any) -> list[str]:
        return _coerce_list(value)


class CareerRecommendation(ATSBaseModel):
    best_matching_roles: list[RoleRecommendation] = Field(default_factory=list)
    salary_band: str = ""
    learning_path: list[str] = Field(default_factory=list)
    missing_technologies: list[str] = Field(default_factory=list)
    portfolio_projects: list[str] = Field(default_factory=list)
    interview_focus: list[str] = Field(default_factory=list)

    @field_validator("learning_path", "missing_technologies", "portfolio_projects", "interview_focus", mode="before")
    @classmethod
    def normalize_lists(cls, value: Any) -> list[str]:
        return _coerce_list(value)


class ATSReport(ATSBaseModel):
    ats_score: int = Field(default=0, ge=0, le=100)
    score_breakdown: ATSScoreBreakdown = Field(default_factory=ATSScoreBreakdown)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    strong_skills: list[str] = Field(default_factory=list)
    recommended_skills: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    career_paths: list[str] = Field(default_factory=list)
    fit_summary: str = ""
    risk_flags: list[str] = Field(default_factory=list)

    @field_validator("ats_score", mode="before")
    @classmethod
    def clamp_total(cls, value: Any) -> int:
        return _clamp_score(value, 100)

    @field_validator(
        "matched_skills",
        "missing_skills",
        "strong_skills",
        "recommended_skills",
        "recommendations",
        "career_paths",
        "risk_flags",
        mode="before",
    )
    @classmethod
    def normalize_lists(cls, value: Any) -> list[str]:
        return _coerce_list(value)
