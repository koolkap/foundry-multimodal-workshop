from __future__ import annotations

import json
import os
from typing import Any

from models.ats_report import ATSReport, CareerRecommendation, JobDescriptionAnalysis, ResumeData


LEARNING_MEDIA_SYSTEM_PROMPT = """
You are an expert technical curriculum designer and career coach.
Create practical, personalized learning content that closes ATS skill gaps.
Use the candidate resume profile and ATS analysis as the source of truth.
Return only valid JSON. Do not include markdown fences, commentary, or prose outside JSON.
""".strip()


LEARNING_MEDIA_USER_PROMPT_TEMPLATE = """
Build personalized learning media for the candidate.

Prioritize the top missing skill as the primary learning skill, then weave in the other
missing skills where they naturally support the roadmap, projects, quiz, and presentation.

Return one JSON object with this exact top-level structure:
{
  "metadata": {
    "target_skill": "",
    "missing_skills": [],
    "ats_score": 0,
    "personalization_summary": ""
  },
  "learning_roadmap": {
    "learning_outcome": "",
    "weeks": [
      {
        "week": 1,
        "title": "",
        "weekly_objectives": [],
        "activities": [],
        "deliverable": ""
      }
    ]
  },
  "lesson_plan": {
    "skill": "",
    "learning_objectives": [],
    "agenda": [],
    "activities": [],
    "assessment": []
  },
  "quiz": {
    "mcqs": [
      {
        "question": "",
        "options": [],
        "answer": "",
        "explanation": ""
      }
    ],
    "short_answer_questions": [
      {
        "question": "",
        "sample_answer": ""
      }
    ]
  },
  "practical_assignment": {
    "project_title": "",
    "problem_statement": "",
    "deliverables": [],
    "evaluation_criteria": []
  },
  "mini_project": {
    "title": "",
    "architecture": [],
    "features": [],
    "tech_stack": [],
    "github_deliverables": []
  },
  "presentation_outline": [
    {
      "slide": 1,
      "title": "",
      "bullet_points": []
    }
  ]
}

Rules:
- learning_roadmap.weeks must contain exactly 8 weekly items.
- quiz.mcqs must contain exactly 5 MCQs with 4 options each.
- quiz.short_answer_questions must contain exactly 3 items.
- presentation_outline must contain exactly 10 slide items.
- Make every section hands-on, specific, and portfolio oriented.
- Keep the level realistic for the candidate based on resume experience and matched skills.
- Use concise text that renders cleanly in a Streamlit application.

Resume and ATS payload JSON:
{{RESUME_ATS_PAYLOAD_JSON}}
""".strip()


def generate_learning_media(
    resume_data: ResumeData,
    ats_report: ATSReport,
    job_analysis: JobDescriptionAnalysis | None = None,
    career_recommendation: CareerRecommendation | None = None,
) -> dict[str, Any]:
    """Generate structured learning media from resume and ATS gap analysis."""
    missing_skills = _learning_skill_gaps(ats_report, career_recommendation)
    payload = _build_prompt_payload(resume_data, ats_report, missing_skills, job_analysis, career_recommendation)
    prompt = LEARNING_MEDIA_USER_PROMPT_TEMPLATE.replace(
        "{{RESUME_ATS_PAYLOAD_JSON}}",
        json.dumps(payload, indent=2, ensure_ascii=False),
    )

    client, deployment = _azure_openai_client()
    messages = [
        {"role": "system", "content": LEARNING_MEDIA_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response = _create_chat_completion(client, deployment, messages)
    content = _response_content(response)
    parsed = _parse_json_object(content)
    return _normalize_learning_media(parsed, missing_skills, ats_report)


def _azure_openai_client() -> tuple[Any, str]:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()

    missing = [
        name
        for name, value in {
            "AZURE_OPENAI_ENDPOINT": endpoint,
            "AZURE_OPENAI_API_KEY": api_key,
            "AZURE_OPENAI_DEPLOYMENT": deployment,
            "AZURE_OPENAI_API_VERSION": api_version,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Azure OpenAI configuration is missing: {', '.join(missing)}.")
    _validate_azure_openai_settings(endpoint, api_key, deployment, api_version)

    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required. Install dependencies from requirements.txt.") from exc

    return (
        AzureOpenAI(
            azure_endpoint=endpoint.rstrip("/"),
            api_key=api_key,
            api_version=api_version,
        ),
        deployment,
    )


def _validate_azure_openai_settings(endpoint: str, api_key: str, deployment: str, api_version: str) -> None:
    placeholder_values = {
        "AZURE_OPENAI_ENDPOINT": endpoint,
        "AZURE_OPENAI_API_KEY": api_key,
        "AZURE_OPENAI_DEPLOYMENT": deployment,
        "AZURE_OPENAI_API_VERSION": api_version,
    }
    placeholders = [name for name, value in placeholder_values.items() if _looks_like_placeholder(value)]
    if placeholders:
        raise ValueError(
            "Azure OpenAI configuration still contains placeholder values: "
            f"{', '.join(placeholders)}. Update .env with your real Azure OpenAI resource endpoint, key, "
            "model deployment name, and API version."
        )
    if not endpoint.lower().startswith("https://"):
        raise ValueError("AZURE_OPENAI_ENDPOINT must start with https://.")
    normalized_endpoint = endpoint.lower()
    supported_endpoint = (
        ".openai.azure.com" in normalized_endpoint
        or ".cognitiveservices.azure.com" in normalized_endpoint
    )
    if not supported_endpoint:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT should look like https://<resource-name>.openai.azure.com/ "
            "or https://<resource-name>.cognitiveservices.azure.com/. Do not use the "
            "Content Understanding .services.ai.azure.com endpoint here."
        )


def _looks_like_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return not normalized or "<" in normalized or ">" in normalized or normalized.startswith("your-")


def _create_chat_completion(client: Any, deployment: str, messages: list[dict[str, str]]) -> Any:
    request = {
        "model": deployment,
        "messages": messages,
        "temperature": 0.35,
        "max_tokens": 5000,
    }
    try:
        return client.chat.completions.create(
            **request,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        if "response_format" not in str(exc).lower():
            raise _friendly_openai_error(exc) from exc
        try:
            return client.chat.completions.create(**request)
        except Exception as retry_exc:
            raise _friendly_openai_error(retry_exc) from retry_exc


def _friendly_openai_error(exc: Exception) -> Exception:
    class_name = exc.__class__.__name__
    message = str(exc).strip() or class_name
    if class_name == "APIConnectionError":
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        return ConnectionError(
            "Could not connect to Azure OpenAI. Check that AZURE_OPENAI_ENDPOINT is the real "
            f"Azure OpenAI resource endpoint, the deployment '{deployment}' exists, and this machine can reach "
            f"{endpoint or 'the configured endpoint'}."
        )
    return RuntimeError(f"Azure OpenAI request failed: {class_name}: {message}")


def _response_content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError) as exc:
        raise ValueError("Azure OpenAI returned an unexpected response shape.") from exc
    if isinstance(content, list):
        return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    if not content:
        raise ValueError("Azure OpenAI returned an empty learning media response.")
    return str(content).strip()


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Learning media response was not valid JSON.") from None
        try:
            parsed = json.loads(content[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("Learning media response was not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Learning media response must be a JSON object.")
    return parsed


def _build_prompt_payload(
    resume_data: ResumeData,
    ats_report: ATSReport,
    missing_skills: list[str],
    job_analysis: JobDescriptionAnalysis | None,
    career_recommendation: CareerRecommendation | None,
) -> dict[str, Any]:
    return {
        "resume_data": {
            "name": resume_data.name,
            "skills": resume_data.skills,
            "experience": [item.model_dump() for item in resume_data.experience],
            "education": [item.model_dump() for item in resume_data.education],
            "certifications": resume_data.certifications,
            "projects": [item.model_dump() for item in resume_data.projects],
            "total_years_experience": resume_data.total_years_experience,
            "professional_summary": resume_data.professional_summary,
        },
        "ats_report": {
            "ats_score": ats_report.ats_score,
            "matched_skills": ats_report.matched_skills,
            "missing_skills": missing_skills,
            "recommendations": ats_report.recommendations,
            "fit_summary": ats_report.fit_summary,
            "risk_flags": ats_report.risk_flags,
        },
        "job_description_analysis": job_analysis.model_dump() if job_analysis else {},
        "career_recommendation": career_recommendation.model_dump() if career_recommendation else {},
    }


def _learning_skill_gaps(
    ats_report: ATSReport,
    career_recommendation: CareerRecommendation | None,
) -> list[str]:
    gaps = [
        *ats_report.missing_skills,
        *ats_report.recommended_skills,
        *(career_recommendation.missing_technologies if career_recommendation else []),
    ]
    deduped = _dedupe(gaps)
    if deduped:
        return deduped[:8]
    return ["Portfolio evidence", "Role-specific project storytelling"]


def _normalize_learning_media(
    media: dict[str, Any],
    missing_skills: list[str],
    ats_report: ATSReport,
) -> dict[str, Any]:
    target_skill = missing_skills[0] if missing_skills else "Skill development"
    metadata = _as_dict(media.get("metadata"))
    metadata.setdefault("target_skill", target_skill)
    metadata.setdefault("missing_skills", missing_skills)
    metadata.setdefault("ats_score", ats_report.ats_score)
    metadata.setdefault("personalization_summary", "")

    roadmap = _as_dict(media.get("learning_roadmap"))
    roadmap.setdefault("learning_outcome", "")
    roadmap["weeks"] = _as_list(roadmap.get("weeks"))[:8]

    lesson_plan = _as_dict(media.get("lesson_plan"))
    lesson_plan.setdefault("skill", target_skill)

    quiz = _as_dict(media.get("quiz"))
    quiz["mcqs"] = _as_list(quiz.get("mcqs"))[:5]
    quiz["short_answer_questions"] = _as_list(quiz.get("short_answer_questions"))[:3]

    return {
        "metadata": metadata,
        "learning_roadmap": roadmap,
        "lesson_plan": lesson_plan,
        "quiz": quiz,
        "practical_assignment": _as_dict(media.get("practical_assignment")),
        "mini_project": _as_dict(media.get("mini_project")),
        "presentation_outline": _as_list(media.get("presentation_outline"))[:10],
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result
