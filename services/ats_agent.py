from __future__ import annotations

import re
from typing import Any

from models.ats_report import (
    ATSReport,
    ATSScoreBreakdown,
    CareerRecommendation,
    EducationEntry,
    ExperienceEntry,
    JobDescriptionAnalysis,
    ProjectEntry,
    ResumeData,
    RoleRecommendation,
    SkillAnalysis,
)


SKILL_CATALOG: dict[str, list[str]] = {
    "programming_languages": [
        "Python",
        "Java",
        "JavaScript",
        "TypeScript",
        "C#",
        "C++",
        "Go",
        "Rust",
        "SQL",
        "PowerShell",
        "Bash",
    ],
    "cloud_skills": [
        "Azure",
        "Azure OpenAI",
        "Azure AI Foundry",
        "Azure AI Search",
        "Azure Functions",
        "Azure Kubernetes Service",
        "AWS",
        "GCP",
        "Cloud Architecture",
    ],
    "devops_skills": [
        "Docker",
        "Kubernetes",
        "Terraform",
        "CI/CD",
        "GitHub Actions",
        "Azure DevOps",
        "Jenkins",
        "Linux",
        "Monitoring",
    ],
    "genai_skills": [
        "Generative AI",
        "GenAI",
        "RAG",
        "LangChain",
        "Prompt Engineering",
        "Vector Database",
        "Embeddings",
        "Agents",
        "LLM",
        "OpenAI",
    ],
    "technical_skills": [
        "React",
        "Node.js",
        "FastAPI",
        "Django",
        "Flask",
        "Streamlit",
        "REST API",
        "GraphQL",
        "Microservices",
        "Machine Learning",
        "Data Engineering",
        "Pandas",
        "PySpark",
        "MongoDB",
        "PostgreSQL",
        "MySQL",
    ],
    "tools": [
        "Git",
        "GitHub",
        "VS Code",
        "Jira",
        "Postman",
        "Figma",
        "Power BI",
        "Tableau",
    ],
    "soft_skills": [
        "Leadership",
        "Communication",
        "Problem Solving",
        "Collaboration",
        "Mentoring",
        "Ownership",
        "Stakeholder Management",
        "Agile",
    ],
}

CERTIFICATION_TERMS = [
    "AZ-900",
    "AI-900",
    "AI-102",
    "AZ-104",
    "AZ-204",
    "AZ-305",
    "AWS Certified",
    "CKA",
    "CKAD",
    "PMP",
    "Scrum Master",
]


class ATSAgent:
    """Local ATS intelligence built around Azure Content Understanding output."""

    def extract_resume_data(
        self,
        resume_text: str,
        *,
        source_file: str = "",
        raw_result: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> ResumeData:
        fields = self._extract_cu_fields(raw_result or {})
        resume = ResumeData(
            name=self._first_text(fields, ["FullName", "name", "candidate_name", "full_name", "person_name"]),
            email=self._first_text(fields, ["ContactEmail", "email", "email_address", "e_mail"]),
            phone=self._first_text(fields, ["ContactPhone", "phone", "phone_number", "mobile", "contact_number"]),
            linkedin=self._first_text(fields, ["linkedin", "linked_in", "linkedin_url", "profile_url"]),
            location=self._first_text(fields, ["location", "address", "city", "current_location"]),
            professional_summary=self._first_text(fields, ["summary", "professional_summary", "profile", "objective"]),
            skills=self._skills_from_fields(fields),
            experience=self._experience_entries(
                self._first_value(
                    fields,
                    ["ProfessionalExperience", "experience", "work_experience", "employment", "employment_history"],
                )
            ),
            education=self._education_entries(self._first_value(fields, ["Education", "academic", "academics"])),
            certifications=self._as_list(
                self._first_value(
                    fields,
                    ["certifications", "certificates", "licenses", "credentials", "PatentsAndRecognition"],
                )
            ),
            projects=self._project_entries(
                self._first_value(fields, ["KeyProductsAndProjects", "projects", "project_experience"])
            ),
            total_years_experience=self._experience_years_from_fields(fields),
            raw_text=resume_text,
            source_file=source_file,
            confidence=self._first_float(fields, ["confidence", "confidence_score"]),
            warnings=warnings or [],
        )
        self._fill_resume_gaps(resume)
        return resume

    def analyze_skills(self, resume: ResumeData) -> SkillAnalysis:
        all_resume_text = " ".join(
            [
                resume.raw_text,
                " ".join(resume.skills),
                " ".join(resume.certifications),
                " ".join(project.description for project in resume.projects),
            ]
        )
        detected = self._dedupe([*resume.skills, *self._detect_catalog_skills(all_resume_text)])
        analysis = SkillAnalysis(
            technical_skills=self._filter_catalog(detected, "technical_skills"),
            cloud_skills=self._filter_catalog(detected, "cloud_skills"),
            devops_skills=self._filter_catalog(detected, "devops_skills"),
            genai_skills=self._filter_catalog(detected, "genai_skills"),
            soft_skills=self._filter_catalog(detected, "soft_skills"),
            tools=self._filter_catalog(detected, "tools"),
            programming_languages=self._filter_catalog(detected, "programming_languages"),
            other_skills=[],
        )
        categorized = self._dedupe(
            [
                *analysis.technical_skills,
                *analysis.cloud_skills,
                *analysis.devops_skills,
                *analysis.genai_skills,
                *analysis.soft_skills,
                *analysis.tools,
                *analysis.programming_languages,
            ]
        )
        analysis.other_skills = [skill for skill in detected if self._normalize(skill) not in self._norm_set(categorized)]
        analysis.all_skills = self._dedupe([*categorized, *analysis.other_skills])
        analysis.skill_distribution = {
            "Technical": len(analysis.technical_skills),
            "Cloud": len(analysis.cloud_skills),
            "DevOps": len(analysis.devops_skills),
            "GenAI": len(analysis.genai_skills),
            "Soft Skills": len(analysis.soft_skills),
            "Tools": len(analysis.tools),
            "Programming": len(analysis.programming_languages),
            "Other": len(analysis.other_skills),
        }
        return analysis

    def generate_ats_report(
        self,
        resume: ResumeData,
        skill_analysis: SkillAnalysis,
        job_analysis: JobDescriptionAnalysis,
        job_description: str,
    ) -> ATSReport:
        del job_description
        resume_skills = self._dedupe([*resume.skills, *skill_analysis.all_skills])
        resume_norm = self._normalized_map(resume_skills)
        required = self._dedupe(job_analysis.required_skills)
        preferred = self._dedupe([*job_analysis.preferred_skills, *job_analysis.soft_skills])
        required_norm = self._normalized_map(required)
        preferred_norm = self._normalized_map(preferred)

        matched_required = sorted(set(required_norm) & set(resume_norm))
        matched_preferred = sorted(set(preferred_norm) & set(resume_norm))
        missing_required = sorted(set(required_norm) - set(resume_norm))
        missing_preferred = sorted(set(preferred_norm) - set(resume_norm))

        skills_score = self._skills_score(required, preferred, matched_required, matched_preferred)
        experience_score = self._experience_score(resume.total_years_experience, job_analysis.years_experience)
        education_score = self._education_score(resume, job_analysis)
        certification_score = self._certification_score(resume.certifications, job_analysis.certifications)
        breakdown = ATSScoreBreakdown(
            skills=skills_score,
            experience=experience_score,
            education=education_score,
            certifications=certification_score,
        )

        matched_skills = self._dedupe(
            [required_norm[key] for key in matched_required] + [preferred_norm[key] for key in matched_preferred]
        )
        missing_skills = self._dedupe(
            [required_norm[key] for key in missing_required] + [preferred_norm[key] for key in missing_preferred]
        )
        recommendations = self._build_recommendations(missing_skills, resume, job_analysis)
        career_paths = self._career_paths_for_skills(skill_analysis)

        return ATSReport(
            ats_score=breakdown.total,
            score_breakdown=breakdown,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            strong_skills=matched_skills[:12],
            recommended_skills=missing_skills[:15],
            recommendations=recommendations,
            career_paths=career_paths,
            fit_summary=self._fit_summary(breakdown.total, matched_skills, missing_skills),
            risk_flags=self._risk_flags(resume, missing_skills, job_analysis),
        )

    def generate_career_recommendations(
        self,
        resume: ResumeData,
        skill_analysis: SkillAnalysis,
        job_analysis: JobDescriptionAnalysis | None = None,
        ats_report: ATSReport | None = None,
    ) -> CareerRecommendation:
        missing = ats_report.missing_skills if ats_report else []
        roles = [
            RoleRecommendation(
                role=role,
                fit_score=self._role_fit(role, skill_analysis, ats_report),
                salary_band=self._salary_band(role),
                rationale=self._role_rationale(role, skill_analysis),
                learning_path=self._learning_path_for_missing(missing),
                missing_technologies=missing[:8],
            )
            for role in self._career_paths_for_skills(skill_analysis)
        ]
        if not roles:
            roles = [
                RoleRecommendation(
                    role="Full Stack Developer",
                    fit_score=max(55, ats_report.ats_score if ats_report else 60),
                    salary_band="$95k-$145k",
                    rationale="General software engineering skills are present.",
                    learning_path=self._learning_path_for_missing(missing),
                    missing_technologies=missing[:8],
                )
            ]
        all_missing = self._dedupe([*missing, *(job_analysis.required_skills if job_analysis else [])])[:10]
        return CareerRecommendation(
            best_matching_roles=roles[:4],
            salary_band=roles[0].salary_band,
            learning_path=self._learning_path_for_missing(all_missing),
            missing_technologies=all_missing,
            portfolio_projects=self._portfolio_projects(all_missing),
            interview_focus=self._interview_focus(skill_analysis, all_missing),
        )

    def _extract_cu_fields(self, raw_result: dict[str, Any]) -> dict[str, Any]:
        candidate_field_sets: list[dict[str, Any]] = []
        self._collect_field_sets(raw_result, candidate_field_sets)
        fields: dict[str, Any] = {}
        for field_set in candidate_field_sets:
            for key, value in field_set.items():
                fields[self._field_key(key)] = self._coerce_cu_value(value)
        return fields

    def _skills_from_fields(self, fields: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for alias in ["skills", "TechnicalSkills", "key_skills", "skill_set"]:
            values.extend(self._as_list(self._first_value(fields, [alias])))

        leadership = self._first_value(fields, ["CoreLeadershipCompetencies"])
        values.extend(self._as_list(leadership))

        detected = self._detect_catalog_skills(" ".join(values))
        return self._dedupe([*values, *detected])

    def _experience_years_from_fields(self, fields: dict[str, Any]) -> float:
        direct_years = self._first_float(
            fields,
            ["total_years_experience", "years_experience", "experience_years", "total_experience"],
        )
        if direct_years:
            return direct_years

        highlights = self._first_value(fields, ["ExperienceHighlights"])
        if isinstance(highlights, dict):
            return self._first_float(
                {self._field_key(key): value for key, value in highlights.items()},
                ["YearsExperience", "total_years_experience", "years_experience"],
            )
        return 0.0

    def _collect_field_sets(self, node: Any, field_sets: list[dict[str, Any]]) -> None:
        if isinstance(node, dict):
            fields = node.get("fields")
            if isinstance(fields, dict):
                field_sets.append(fields)
            for value in node.values():
                self._collect_field_sets(value, field_sets)
        elif isinstance(node, list):
            for item in node:
                self._collect_field_sets(item, field_sets)

    def _coerce_cu_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            for key in (
                "valueString",
                "valueDate",
                "valuePhoneNumber",
                "valueNumber",
                "valueInteger",
                "valueBoolean",
                "content",
                "text",
                "markdown",
                "value",
            ):
                if key in value and value[key] not in (None, ""):
                    return self._coerce_cu_value(value[key])
            if isinstance(value.get("valueArray"), list):
                return [self._coerce_cu_value(item) for item in value["valueArray"]]
            if isinstance(value.get("valueObject"), dict):
                return {
                    self._field_key(key): self._coerce_cu_value(item)
                    for key, item in value["valueObject"].items()
                }
            if isinstance(value.get("fields"), dict):
                return {
                    self._field_key(key): self._coerce_cu_value(item)
                    for key, item in value["fields"].items()
                }
            return {
                self._field_key(key): self._coerce_cu_value(item)
                for key, item in value.items()
                if key not in {"type", "spans", "confidence", "source"}
            }
        if isinstance(value, list):
            return [self._coerce_cu_value(item) for item in value]
        return value

    def _fill_resume_gaps(self, resume: ResumeData) -> None:
        text = resume.raw_text
        if not resume.email:
            match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
            resume.email = match.group(0) if match else ""
        if not resume.phone:
            match = re.search(r"(?:\+?\d[\d\s().-]{7,}\d)", text)
            resume.phone = match.group(0).strip() if match else ""
        if not resume.linkedin:
            match = re.search(r"https?://(?:www\.)?linkedin\.com/[^\s)]+", text, flags=re.IGNORECASE)
            resume.linkedin = match.group(0) if match else ""
        if not resume.skills:
            resume.skills = self._detect_catalog_skills(text)
        if not resume.certifications:
            resume.certifications = [cert for cert in CERTIFICATION_TERMS if self._contains_term(text, cert)]
        if resume.total_years_experience <= 0:
            resume.total_years_experience = self._estimate_years_experience(text)
        if not resume.name:
            resume.name = self._guess_name(text)

    def _first_value(self, fields: dict[str, Any], aliases: list[str]) -> Any:
        compact_fields = {self._compact_key(key): value for key, value in fields.items()}
        for alias in aliases:
            normalized = self._field_key(alias)
            if normalized in fields:
                return fields[normalized]
            compact = self._compact_key(alias)
            if compact in compact_fields:
                return compact_fields[compact]
        return None

    def _first_text(self, fields: dict[str, Any], aliases: list[str]) -> str:
        value = self._first_value(fields, aliases)
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(item) for item in value if item)
        if isinstance(value, dict):
            return ", ".join(str(item) for item in value.values() if item)
        return str(value).strip()

    def _first_float(self, fields: dict[str, Any], aliases: list[str]) -> float:
        value = self._first_value(fields, aliases)
        if isinstance(value, str):
            match = re.search(r"\d+(?:\.\d+)?", value)
            value = match.group(0) if match else value
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0

    def _experience_entries(self, value: Any) -> list[ExperienceEntry]:
        items = self._as_records(value)
        entries: list[ExperienceEntry] = []
        for item in items:
            entries.append(
                ExperienceEntry(
                    title=self._record_text(item, ["title", "role", "position", "job_title"]),
                    company=self._record_text(item, ["company", "employer", "organization"]),
                    location=self._record_text(item, ["location"]),
                    start_date=self._record_text(item, ["start_date", "from"]),
                    end_date=self._record_text(item, ["end_date", "to", "present"]),
                    duration=self._record_text(item, ["duration"]),
                    responsibilities=self._as_list(
                        self._record_value(item, ["responsibilities", "description", "achievements"])
                    ),
                    technologies=self._as_list(self._record_value(item, ["technologies", "tech_stack", "skills"])),
                )
            )
        return [entry for entry in entries if entry.title or entry.company or entry.responsibilities]

    def _education_entries(self, value: Any) -> list[EducationEntry]:
        items = self._as_records(value)
        entries: list[EducationEntry] = []
        for item in items:
            entries.append(
                EducationEntry(
                    institution=self._record_text(item, ["institution", "school", "college", "university"]),
                    degree=self._record_text(item, ["degree", "qualification"]),
                    field_of_study=self._record_text(item, ["field_of_study", "major", "specialization"]),
                    graduation_year=self._record_text(item, ["graduation_year", "year", "end_date"]),
                    location=self._record_text(item, ["location"]),
                    achievements=self._as_list(self._record_value(item, ["achievements", "honors", "GpaOrScore"])),
                )
            )
        return [entry for entry in entries if entry.institution or entry.degree]

    def _project_entries(self, value: Any) -> list[ProjectEntry]:
        items = self._as_records(value)
        entries: list[ProjectEntry] = []
        for item in items:
            entries.append(
                ProjectEntry(
                    name=self._record_text(item, ["ProductName", "name", "project", "title"]),
                    description=self._record_text(item, ["description", "summary"]),
                    technologies=self._as_list(self._record_value(item, ["technologies", "tech_stack", "skills"])),
                    outcomes=self._as_list(
                        self._record_value(item, ["outcomes", "impact", "results", "ProjectDates"])
                    ),
                )
            )
        return [entry for entry in entries if entry.name or entry.description]

    def _as_records(self, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, dict):
            if any(isinstance(item, dict) for item in value.values()):
                return [item for item in value.values() if isinstance(item, dict)]
            return [value]
        if isinstance(value, list):
            records: list[dict[str, Any]] = []
            for item in value:
                if isinstance(item, dict):
                    records.append(item)
                elif item:
                    records.append({"description": item})
            return records
        return [{"description": value}]

    def _record_value(self, record: dict[str, Any], aliases: list[str]) -> Any:
        normalized = {self._field_key(key): value for key, value in record.items()}
        return self._first_value(normalized, aliases)

    def _record_text(self, record: dict[str, Any], aliases: list[str]) -> str:
        value = self._record_value(record, aliases)
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(item) for item in value if item)
        if isinstance(value, dict):
            return ", ".join(str(item) for item in value.values() if item)
        return str(value).strip()

    def _as_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            result: list[str] = []
            for item in value:
                result.extend(self._as_list(item))
            return self._dedupe(result)
        if isinstance(value, dict):
            result: list[str] = []
            for item in value.values():
                result.extend(self._as_list(item))
            return self._dedupe(result)
        if isinstance(value, str):
            pieces = re.split(r"[,;\n|]+", value)
            return self._dedupe([piece.strip(" -\t") for piece in pieces if piece.strip(" -\t")])
        return [str(value).strip()] if str(value).strip() else []

    def _detect_catalog_skills(self, text: str) -> list[str]:
        detected: list[str] = []
        for values in SKILL_CATALOG.values():
            for skill in values:
                if self._contains_term(text, skill):
                    detected.append(skill)
        return self._dedupe(detected)

    def _filter_catalog(self, detected: list[str], category: str) -> list[str]:
        catalog_norm = self._norm_set(SKILL_CATALOG[category])
        return [skill for skill in detected if self._normalize(skill) in catalog_norm]

    def _contains_term(self, text: str, term: str) -> bool:
        pattern = r"(?<![a-zA-Z0-9+#.])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![a-zA-Z0-9+#.])"
        return bool(re.search(pattern, text, flags=re.IGNORECASE))

    def _estimate_years_experience(self, text: str) -> float:
        matches = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s+(?:of\s+)?experience", text, re.IGNORECASE)
        if matches:
            return max(float(match) for match in matches)
        years = [int(year) for year in re.findall(r"\b(20\d{2}|19\d{2})\b", text)]
        if len(years) >= 2:
            return max(0.0, min(30.0, float(max(years) - min(years))))
        return 0.0

    def _guess_name(self, text: str) -> str:
        for line in text.splitlines()[:8]:
            cleaned = line.strip()
            if not cleaned or "@" in cleaned or re.search(r"\d", cleaned):
                continue
            words = cleaned.split()
            if 2 <= len(words) <= 4 and all(word[:1].isupper() for word in words):
                return cleaned
        return ""

    def _skills_score(
        self,
        required: list[str],
        preferred: list[str],
        matched_required: list[str],
        matched_preferred: list[str],
    ) -> int:
        if required and preferred:
            required_ratio = len(matched_required) / max(1, len(required))
            preferred_ratio = len(matched_preferred) / max(1, len(preferred))
            return round(40 * ((required_ratio * 0.75) + (preferred_ratio * 0.25)))
        if required:
            return round(40 * len(matched_required) / max(1, len(required)))
        if preferred:
            return round(40 * len(matched_preferred) / max(1, len(preferred)))
        return 30

    def _experience_score(self, resume_years: float, required_years: float) -> int:
        if required_years <= 0:
            return 30 if resume_years > 0 else 18
        return round(30 * min(1.0, resume_years / required_years))

    def _education_score(self, resume: ResumeData, job_analysis: JobDescriptionAnalysis) -> int:
        if not resume.education:
            return 0 if job_analysis.education_requirements else 12
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
        resume_norm = self._norm_set(resume_certifications)
        required_norm = self._normalized_map(required_certifications)
        matched = set(required_norm) & resume_norm
        return round(10 * len(matched) / max(1, len(required_norm)))

    def _build_recommendations(
        self,
        missing_skills: list[str],
        resume: ResumeData,
        job_analysis: JobDescriptionAnalysis,
    ) -> list[str]:
        recommendations = [
            f"Add measurable project or work evidence for {skill}."
            for skill in missing_skills[:5]
        ]
        if job_analysis.years_experience and resume.total_years_experience < job_analysis.years_experience:
            recommendations.append(
                "Highlight internships, freelance work, projects, or leadership scope to strengthen experience evidence."
            )
        if job_analysis.certifications and not resume.certifications:
            recommendations.append("Add relevant certifications or list certification plans in a learning section.")
        if not resume.projects:
            recommendations.append("Add 2-3 outcome-focused projects with technologies, metrics, and business impact.")
        return recommendations

    def _career_paths_for_skills(self, analysis: SkillAnalysis) -> list[str]:
        paths: list[str] = []
        if analysis.genai_skills:
            paths.append("AI Engineer")
        if analysis.cloud_skills and analysis.devops_skills:
            paths.append("Cloud Architect")
        if analysis.programming_languages and analysis.technical_skills:
            paths.append("Full Stack Developer")
        if analysis.soft_skills and any(skill in analysis.soft_skills for skill in ["Leadership", "Mentoring"]):
            paths.append("Engineering Manager")
        return self._dedupe(paths)

    def _role_fit(self, role: str, analysis: SkillAnalysis, report: ATSReport | None) -> int:
        base = report.ats_score if report else 65
        boosts = {
            "AI Engineer": len(analysis.genai_skills) * 4,
            "Cloud Architect": len(analysis.cloud_skills) * 3 + len(analysis.devops_skills) * 2,
            "Full Stack Developer": len(analysis.programming_languages) * 3 + len(analysis.technical_skills) * 2,
            "Engineering Manager": len(analysis.soft_skills) * 3,
        }
        return max(0, min(100, base + boosts.get(role, 0)))

    def _salary_band(self, role: str) -> str:
        return {
            "AI Engineer": "$120k-$190k",
            "Cloud Architect": "$135k-$210k",
            "Full Stack Developer": "$100k-$165k",
            "Engineering Manager": "$150k-$240k",
        }.get(role, "$95k-$150k")

    def _role_rationale(self, role: str, analysis: SkillAnalysis) -> str:
        if role == "AI Engineer":
            return "The resume shows GenAI, language model, or AI workflow evidence."
        if role == "Cloud Architect":
            return "The resume combines cloud platform and infrastructure delivery skills."
        if role == "Engineering Manager":
            return "The resume includes leadership or collaboration signals."
        return f"The resume includes {len(analysis.programming_languages)} programming and {len(analysis.technical_skills)} technical skills."

    def _learning_path_for_missing(self, missing: list[str]) -> list[str]:
        if not missing:
            return ["Add metrics to resume bullets", "Prepare role-specific interview stories", "Polish portfolio evidence"]
        return [f"Build hands-on evidence for {skill}" for skill in missing[:6]]

    def _portfolio_projects(self, missing: list[str]) -> list[str]:
        if any(self._normalize(skill) in {"rag", "langchain", "azureopenai", "genai"} for skill in missing):
            return ["Build a RAG resume or knowledge-base assistant with Azure AI Search and Azure OpenAI"]
        if any(self._normalize(skill) in {"docker", "kubernetes", "terraform"} for skill in missing):
            return ["Deploy a containerized app with CI/CD, IaC, monitoring, and rollback notes"]
        return ["Create a job-matching dashboard with resume parsing, scoring, and exportable reports"]

    def _interview_focus(self, analysis: SkillAnalysis, missing: list[str]) -> list[str]:
        focus = ["Explain projects with impact metrics", "Prepare system design tradeoffs"]
        if analysis.genai_skills:
            focus.append("Discuss retrieval quality, grounding, evaluation, and safety")
        if missing:
            focus.append("Prepare honest gap-bridging answers for missing JD skills")
        return focus

    def _fit_summary(self, score: int, matched: list[str], missing: list[str]) -> str:
        if score >= 80:
            return f"Strong ATS fit with {len(matched)} matched skills and limited gaps."
        if score >= 60:
            return f"Moderate ATS fit with {len(matched)} matched skills and {len(missing)} visible gaps."
        return f"Low-to-moderate ATS fit. Address {len(missing)} missing skills before applying."

    def _risk_flags(
        self,
        resume: ResumeData,
        missing: list[str],
        job_analysis: JobDescriptionAnalysis,
    ) -> list[str]:
        flags: list[str] = []
        if missing:
            flags.append("Missing required or preferred JD skills.")
        if job_analysis.years_experience and resume.total_years_experience < job_analysis.years_experience:
            flags.append("Resume years of experience may be below the JD requirement.")
        if job_analysis.certifications and not resume.certifications:
            flags.append("JD mentions certifications that are not visible in the resume.")
        return flags

    def _normalized_map(self, values: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for value in values:
            normalized = self._normalize(value)
            if normalized and normalized not in result:
                result[normalized] = value.strip()
        return result

    def _norm_set(self, values: list[str]) -> set[str]:
        return {self._normalize(value) for value in values if self._normalize(value)}

    def _normalize(self, value: str) -> str:
        normalized = str(value).lower().strip()
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

    def _field_key(self, value: str) -> str:
        key = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
        return key

    def _compact_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            cleaned = str(value).strip()
            normalized = self._normalize(cleaned)
            if not cleaned or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(cleaned)
        return result
