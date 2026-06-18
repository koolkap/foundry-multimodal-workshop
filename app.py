from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime
from html import escape
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from models.ats_report import ATSReport, CareerRecommendation, JobDescriptionAnalysis, ResumeData, SkillAnalysis
from services.ats_agent import ATSAgent
from services.document_processor import ContentUnderstandingSettings, DocumentProcessor
from services.jd_matcher import JDMatcher
from services.learning_media import generate_learning_media


load_dotenv()


def main() -> None:
    st.set_page_config(
        page_title="ATS Resume Intelligence Agent",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_state()
    content_settings, dark_mode = _render_sidebar()
    _inject_theme(dark_mode)

    st.title("ATS Resume Intelligence Agent")

    upload_col, jd_col = st.columns([0.9, 1.1], vertical_alignment="top")
    with upload_col:
        resume_file = st.file_uploader("Resume Upload", type=["pdf", "docx"])
    with jd_col:
        job_description = st.text_area("Job Description Input", height=210)

    action_col_1, action_col_2, action_col_3, action_col_4 = st.columns(4)
    extract_clicked = action_col_1.button("Extract Resume", use_container_width=True)
    analyze_clicked = action_col_2.button("Analyze ATS Score", use_container_width=True)
    recommend_clicked = action_col_3.button("Generate Recommendations", use_container_width=True)
    learning_clicked = action_col_4.button("Generate Learning Media", use_container_width=True)

    if extract_clicked:
        _run_resume_extraction(resume_file, content_settings)

    if analyze_clicked:
        if _ensure_resume_available(resume_file, content_settings):
            _run_ats_analysis(job_description)

    if recommend_clicked:
        if _ensure_resume_available(resume_file, content_settings):
            _run_recommendations()

    if learning_clicked:
        if _ensure_resume_available(resume_file, content_settings):
            _run_learning_media()

    resume: ResumeData | None = st.session_state.get("resume_data")
    skill_analysis: SkillAnalysis | None = st.session_state.get("skill_analysis")
    job_analysis: JobDescriptionAnalysis | None = st.session_state.get("job_analysis")
    ats_report: ATSReport | None = st.session_state.get("ats_report")
    career_recommendation: CareerRecommendation | None = st.session_state.get("career_recommendation")
    learning_media: dict[str, Any] | None = st.session_state.get("learning_media")

    if resume:
        _render_resume_tabs(resume, skill_analysis)

    if ats_report:
        _render_dashboard(ats_report, skill_analysis, career_recommendation)
        if learning_media:
            _render_learning_media(learning_media)
        _render_exports(resume, skill_analysis, job_analysis, ats_report, career_recommendation, learning_media)

    _render_history()


def _init_session_state() -> None:
    defaults = {
        "resume_data": None,
        "skill_analysis": None,
        "job_analysis": None,
        "ats_report": None,
        "career_recommendation": None,
        "learning_media": None,
        "document_source": "",
        "history": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _render_sidebar() -> tuple[ContentUnderstandingSettings, bool]:
    env_content = ContentUnderstandingSettings.from_env()

    with st.sidebar:
        st.header("Content Understanding")
        content_endpoint = st.text_input("Endpoint", value=env_content.endpoint)
        use_content_key_override = st.toggle(
            "Manual key override",
            value=False,
            help="Keep this off to use CONTENTUNDERSTANDING_KEY from .env or DefaultAzureCredential.",
            key="manual_content_key_override_v3",
        )
        if use_content_key_override:
            content_key_input = st.text_input(
                "Content Understanding Key Override",
                value="",
                type="password",
                placeholder="Optional key",
                key="content_understanding_key_override_v3",
            )
        else:
            content_key_input = ""
            if env_content.key:
                st.caption("Content Understanding key is loaded from .env.")
            else:
                st.caption("Content Understanding key is blank; DefaultAzureCredential will be used.")
        analyzer_id = st.text_input("Analyzer ID", value=env_content.analyzer_id)
        content_api_version = st.text_input("API Version", value=env_content.api_version)

        dark_mode = st.toggle("Dark Mode", value=False)

    content_settings = ContentUnderstandingSettings(
        endpoint=content_endpoint,
        key=content_key_input.strip() or env_content.key,
        analyzer_id=analyzer_id,
        api_version=content_api_version,
    )
    with st.sidebar:
        _render_configuration_status(content_settings)
    return content_settings, dark_mode


def _render_configuration_status(content_settings: ContentUnderstandingSettings) -> None:
    errors = _configuration_errors(content_settings)
    if errors:
        for error in errors:
            st.error(error)
        return
    st.success("Content Understanding configuration is ready.")


def _configuration_errors(content_settings: ContentUnderstandingSettings) -> list[str]:
    errors: list[str] = []
    content_endpoint = content_settings.endpoint.strip()

    if not content_endpoint:
        errors.append("Content Understanding endpoint is required.")
    elif not _is_foundry_services_endpoint(content_endpoint):
        errors.append(
            "Content Understanding endpoint should end with .services.ai.azure.com, "
            "for example https://<resource-name>.services.ai.azure.com/."
        )
    if not content_settings.analyzer_id.strip():
        errors.append("Analyzer ID is required. Use the analyzer ID created in Content Understanding Studio, such as ats.")
    return errors


def _normalize_endpoint(endpoint: str) -> str:
    return endpoint.strip().rstrip("/").lower()


def _is_foundry_services_endpoint(endpoint: str) -> bool:
    normalized = _normalize_endpoint(endpoint)
    return normalized.startswith("https://") and normalized.endswith(".services.ai.azure.com")


def _inject_theme(dark_mode: bool) -> None:
    if dark_mode:
        colors = {
            "app_bg": "#0f1419",
            "sidebar_bg": "#151b22",
            "surface": "#1b232c",
            "surface_alt": "#202a34",
            "input_bg": "#111820",
            "text": "#f4f7fb",
            "heading": "#f8fafc",
            "muted": "#a9b4c0",
            "accent": "#14a99a",
            "accent_hover": "#0f8f83",
            "border": "#32404d",
            "button_text": "#ffffff",
            "danger": "#ffb4a8",
        }
    else:
        colors = {
            "app_bg": "#f6f8fb",
            "sidebar_bg": "#ffffff",
            "surface": "#ffffff",
            "surface_alt": "#eef3f7",
            "input_bg": "#ffffff",
            "text": "#17212b",
            "heading": "#101820",
            "muted": "#647180",
            "accent": "#0f8b8d",
            "accent_hover": "#0b7375",
            "border": "#d6dee8",
            "button_text": "#ffffff",
            "danger": "#b42318",
        }

    st.markdown(
        f"""
        <style>
        :root {{
            --ats-bg: {colors["app_bg"]};
            --ats-sidebar: {colors["sidebar_bg"]};
            --ats-surface: {colors["surface"]};
            --ats-surface-alt: {colors["surface_alt"]};
            --ats-input: {colors["input_bg"]};
            --ats-text: {colors["text"]};
            --ats-heading: {colors["heading"]};
            --ats-muted: {colors["muted"]};
            --ats-accent: {colors["accent"]};
            --ats-accent-hover: {colors["accent_hover"]};
            --ats-border: {colors["border"]};
            --ats-button-text: {colors["button_text"]};
            --ats-danger: {colors["danger"]};
        }}

        .stApp {{
            background: var(--ats-bg);
            color: var(--ats-text);
        }}

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {{
            background: var(--ats-bg);
            color: var(--ats-text);
        }}

        [data-testid="stHeader"] {{
            background: var(--ats-bg);
            border-bottom: 1px solid var(--ats-border);
        }}

        [data-testid="stSidebar"] {{
            background: var(--ats-sidebar);
            border-right: 1px solid var(--ats-border);
        }}

        [data-testid="stSidebar"] > div:first-child {{
            background: var(--ats-sidebar);
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: var(--ats-heading) !important;
            letter-spacing: 0;
        }}

        p, label, span, div, small {{
            letter-spacing: 0;
        }}

        p,
        label,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] label,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span {{
            color: var(--ats-text) !important;
        }}

        [data-testid="stCaptionContainer"],
        small,
        .muted {{
            color: var(--ats-muted) !important;
        }}

        [data-baseweb="input"],
        [data-baseweb="textarea"],
        [data-baseweb="select"],
        [data-testid="stTextInputRootElement"],
        [data-testid="stTextAreaRootElement"] {{
            background: var(--ats-input) !important;
            border: 1px solid var(--ats-border) !important;
            border-radius: 8px !important;
            color: var(--ats-text) !important;
        }}

        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        input,
        textarea {{
            background: var(--ats-input) !important;
            color: var(--ats-text) !important;
            caret-color: var(--ats-accent) !important;
            border-color: var(--ats-border) !important;
        }}

        [data-testid="stSidebar"] input[type="password"] + div,
        [data-testid="stSidebar"] [data-baseweb="input"] button {{
            display: none !important;
        }}

        input::placeholder,
        textarea::placeholder {{
            color: var(--ats-muted) !important;
            opacity: 1 !important;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            background: var(--ats-surface) !important;
            border: 1px dashed var(--ats-border) !important;
            border-radius: 8px !important;
        }}

        [data-testid="stFileUploaderDropzone"] * {{
            color: var(--ats-text) !important;
        }}

        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] span {{
            color: var(--ats-muted) !important;
        }}

        [data-testid="stFileUploaderDropzone"] button {{
            background: var(--ats-surface-alt) !important;
            border: 1px solid var(--ats-border) !important;
            color: var(--ats-text) !important;
        }}

        [data-testid="stFileUploaderDropzone"] button * {{
            color: var(--ats-text) !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
        }}

        .stTabs [data-baseweb="tab"] {{
            border: 1px solid var(--ats-border);
            border-radius: 6px;
            padding: 8px 12px;
            background: var(--ats-surface);
            color: var(--ats-text);
        }}

        .stTabs [aria-selected="true"] {{
            border-color: var(--ats-accent) !important;
            color: var(--ats-accent) !important;
        }}

        [data-testid="stExpander"] details {{
            background: var(--ats-surface);
            border: 1px solid var(--ats-border);
            border-radius: 8px;
        }}

        [data-testid="stExpander"] summary {{
            background: var(--ats-surface-alt);
            color: var(--ats-text) !important;
            border-radius: 7px 7px 0 0;
        }}

        [data-testid="stExpander"] summary * {{
            color: var(--ats-text) !important;
        }}

        div[data-testid="stMetric"] {{
            background: var(--ats-surface);
            border: 1px solid var(--ats-border);
            border-radius: 8px;
            padding: 14px 16px;
        }}

        div[data-testid="stMetric"] * {{
            color: var(--ats-text) !important;
        }}

        .stButton > button, .stDownloadButton > button {{
            border-radius: 6px;
            border: 1px solid var(--ats-accent) !important;
            background: var(--ats-accent) !important;
            color: var(--ats-button-text) !important;
            font-weight: 600;
            min-height: 44px;
        }}

        .stButton > button *,
        .stDownloadButton > button * {{
            color: var(--ats-button-text) !important;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            background: var(--ats-accent-hover) !important;
            border-color: var(--ats-accent-hover) !important;
            color: var(--ats-button-text) !important;
        }}

        .stButton > button:disabled,
        .stDownloadButton > button:disabled {{
            background: var(--ats-surface-alt) !important;
            border-color: var(--ats-border) !important;
            color: var(--ats-muted) !important;
        }}

        .stButton > button:disabled *,
        .stDownloadButton > button:disabled * {{
            color: var(--ats-muted) !important;
        }}

        [data-testid="stAlert"] {{
            border-radius: 8px;
        }}

        [data-testid="stDataFrame"],
        [data-testid="stJson"] {{
            background: var(--ats-surface);
            border-radius: 8px;
        }}

        a {{
            color: var(--ats-accent) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _run_resume_extraction(
    resume_file: Any,
    content_settings: ContentUnderstandingSettings,
) -> bool:
    if resume_file is None:
        st.warning("Upload a PDF or DOCX resume first.")
        return False
    if errors := _configuration_errors(content_settings):
        for error in errors:
            st.error(error)
        return False

    try:
        with st.spinner("Extracting resume with Azure Content Understanding..."):
            processor = DocumentProcessor(content_settings)
            document_result = processor.process_uploaded_file(resume_file)
            if not document_result.text.strip():
                st.error("No readable text was extracted from the resume.")
                return False

            agent = ATSAgent()
            resume = agent.extract_resume_data(
                document_result.text,
                source_file=document_result.file_name,
                raw_result=document_result.raw_result,
                warnings=document_result.warnings,
            )
            st.session_state.resume_data = resume
            st.session_state.document_source = document_result.source
            st.session_state.skill_analysis = None
            st.session_state.job_analysis = None
            st.session_state.ats_report = None
            st.session_state.career_recommendation = None
            st.session_state.learning_media = None

        st.success(f"Resume extracted through {st.session_state.document_source}.")
        for warning in resume.warnings:
            st.info(warning)
        return True
    except Exception as exc:  # noqa: BLE001 - Streamlit should display recoverable service errors.
        st.error(f"Resume extraction failed: {_format_service_error(exc)}")
        return False


def _ensure_resume_available(
    resume_file: Any,
    content_settings: ContentUnderstandingSettings,
) -> bool:
    if st.session_state.get("resume_data"):
        return True
    return _run_resume_extraction(resume_file, content_settings)


def _run_ats_analysis(job_description: str) -> None:
    if not job_description.strip():
        st.warning("Paste a job description before running ATS analysis.")
        return

    resume: ResumeData = st.session_state.resume_data
    try:
        with st.spinner("Analyzing skills, job requirements, and ATS score locally..."):
            agent = ATSAgent()
            matcher = JDMatcher()
            skill_analysis = agent.analyze_skills(resume)
            job_analysis = matcher.analyze(job_description)
            ats_report = agent.generate_ats_report(resume, skill_analysis, job_analysis, job_description)
            career_recommendation = agent.generate_career_recommendations(
                resume,
                skill_analysis,
                job_analysis,
                ats_report,
            )
            ats_report.career_paths = [
                role.role for role in career_recommendation.best_matching_roles if role.role
            ] or ats_report.career_paths

            st.session_state.skill_analysis = skill_analysis
            st.session_state.job_analysis = job_analysis
            st.session_state.ats_report = ats_report
            st.session_state.career_recommendation = career_recommendation
            st.session_state.learning_media = None
            _add_history(resume, ats_report, job_analysis)
        st.success("ATS analysis complete.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"ATS analysis failed: {_format_service_error(exc)}")


def _run_recommendations() -> None:
    resume: ResumeData = st.session_state.resume_data
    try:
        with st.spinner("Generating career recommendations locally..."):
            agent = ATSAgent()
            skill_analysis = st.session_state.skill_analysis or agent.analyze_skills(resume)
            st.session_state.skill_analysis = skill_analysis
            career_recommendation = agent.generate_career_recommendations(
                resume,
                skill_analysis,
                st.session_state.job_analysis,
                st.session_state.ats_report,
            )
            if st.session_state.ats_report:
                st.session_state.ats_report.career_paths = [
                    role.role for role in career_recommendation.best_matching_roles if role.role
                ]
            st.session_state.career_recommendation = career_recommendation
        st.success("Recommendations generated.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Recommendation generation failed: {_format_service_error(exc)}")


def _run_learning_media() -> None:
    resume: ResumeData | None = st.session_state.get("resume_data")
    ats_report: ATSReport | None = st.session_state.get("ats_report")
    if resume is None:
        st.warning("Extract a resume before generating learning media.")
        return
    if ats_report is None:
        st.warning("Run ATS analysis before generating learning media.")
        return

    if not ats_report.missing_skills and not ats_report.recommended_skills:
        st.info("No missing skills were detected, so the media will focus on portfolio evidence and role alignment.")

    try:
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "configured deployment")
        with st.spinner(f"Generating personalized learning media with Azure OpenAI deployment {deployment}..."):
            st.session_state.learning_media = generate_learning_media(
                resume,
                ats_report,
                st.session_state.get("job_analysis"),
                st.session_state.get("career_recommendation"),
            )
        st.success("Learning media generated.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Learning media generation failed: {_format_service_error(exc)}")


def _format_service_error(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def _render_resume_tabs(resume: ResumeData, skill_analysis: SkillAnalysis | None) -> None:
    st.subheader("Extracted Resume Information")
    overview_tab, skills_tab, experience_tab, education_tab = st.tabs(
        ["Resume Overview", "Skills", "Experience", "Education"]
    )

    with overview_tab:
        col_1, col_2, col_3, col_4 = st.columns(4)
        col_1.metric("Candidate", resume.name or "Unknown")
        col_2.metric("Experience", f"{resume.total_years_experience:g} years")
        col_3.metric("Certifications", len(resume.certifications))
        col_4.metric("Projects", len(resume.projects))

        contact = {
            "Email": resume.email,
            "Phone": resume.phone,
            "LinkedIn": resume.linkedin,
            "Location": resume.location,
            "Document Source": st.session_state.get("document_source", ""),
        }
        st.dataframe(_dict_to_frame(contact), use_container_width=True, hide_index=True)
        if resume.professional_summary:
            st.write(resume.professional_summary)
        st.json(resume.model_dump(exclude={"raw_text"}))

    with skills_tab:
        if skill_analysis:
            groups = {
                "Technical Skills": skill_analysis.technical_skills,
                "Programming Languages": skill_analysis.programming_languages,
                "Cloud Skills": skill_analysis.cloud_skills,
                "DevOps Skills": skill_analysis.devops_skills,
                "GenAI Skills": skill_analysis.genai_skills,
                "Soft Skills": skill_analysis.soft_skills,
                "Tools": skill_analysis.tools,
                "Other Skills": skill_analysis.other_skills,
            }
            for label, values in groups.items():
                if values:
                    st.markdown(f"**{label}**")
                    st.write(", ".join(values))
        else:
            st.write(", ".join(resume.skills) if resume.skills else "No skills extracted.")

    with experience_tab:
        if resume.experience:
            exp_rows = [
                {
                    "Title": item.title,
                    "Company": item.company,
                    "Location": item.location,
                    "Start": item.start_date,
                    "End": item.end_date,
                    "Duration": item.duration,
                    "Technologies": ", ".join(item.technologies),
                }
                for item in resume.experience
            ]
            st.dataframe(pd.DataFrame(exp_rows), use_container_width=True, hide_index=True)
            for item in resume.experience:
                with st.expander(f"{item.title or 'Role'} at {item.company or 'Company'}"):
                    for responsibility in item.responsibilities:
                        st.write(f"- {responsibility}")
        else:
            st.write("No experience entries extracted.")

    with education_tab:
        if resume.education:
            edu_rows = [item.model_dump() for item in resume.education]
            st.dataframe(pd.DataFrame(edu_rows), use_container_width=True, hide_index=True)
        else:
            st.write("No education entries extracted.")
        if resume.certifications:
            st.markdown("**Certifications**")
            st.write(", ".join(resume.certifications))


def _render_dashboard(
    ats_report: ATSReport,
    skill_analysis: SkillAnalysis | None,
    career_recommendation: CareerRecommendation | None,
) -> None:
    st.subheader("ATS Dashboard")
    gauge_col, radar_col = st.columns(2)
    with gauge_col:
        st.plotly_chart(_ats_gauge(ats_report.ats_score), use_container_width=True)
    with radar_col:
        st.plotly_chart(_score_radar(ats_report), use_container_width=True)

    pie_col, missing_col = st.columns([0.9, 1.1])
    with pie_col:
        if skill_analysis:
            st.plotly_chart(_skill_pie(skill_analysis), use_container_width=True)
    with missing_col:
        st.markdown("**Missing Skills Table**")
        missing_rows = [{"Missing Skill": skill} for skill in ats_report.missing_skills]
        st.dataframe(pd.DataFrame(missing_rows), use_container_width=True, hide_index=True)

    rec_rows = [{"Recommendation": item} for item in ats_report.recommendations]
    st.markdown("**Recommendations Table**")
    st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)

    if career_recommendation:
        role_rows = [
            {
                "Role": role.role,
                "Fit Score": role.fit_score,
                "Salary Band": role.salary_band,
                "Missing Technologies": ", ".join(role.missing_technologies),
            }
            for role in career_recommendation.best_matching_roles
        ]
        st.markdown("**Career Paths**")
        st.dataframe(pd.DataFrame(role_rows), use_container_width=True, hide_index=True)


def _render_learning_media(media: dict[str, Any]) -> None:
    metadata = _safe_dict(media.get("metadata"))
    missing_skills = [str(item) for item in _display_list(metadata.get("missing_skills")) if str(item).strip()]
    target_skill = str(metadata.get("target_skill") or (missing_skills[0] if missing_skills else "Skill development"))
    ats_score = metadata.get("ats_score", 0)

    st.subheader("Learning Media Generator")
    summary_col, score_col = st.columns([0.78, 0.22], vertical_alignment="top")
    with summary_col:
        st.markdown(f"**Primary Skill:** {target_skill}")
        if missing_skills:
            st.markdown(f"**Skill Gaps:** {', '.join(missing_skills)}")
        if metadata.get("personalization_summary"):
            st.markdown(str(metadata["personalization_summary"]))
    with score_col:
        st.metric("ATS Score", f"{ats_score}/100")

    roadmap_tab, lesson_tab, quiz_tab, assignment_tab, project_tab, presentation_tab = st.tabs(
        ["Learning Roadmap", "Lesson Plan", "Quiz", "Assignment", "Mini Project", "Presentation"]
    )

    with roadmap_tab:
        roadmap = _safe_dict(media.get("learning_roadmap"))
        if roadmap.get("learning_outcome"):
            st.markdown("**Learning Outcome**")
            st.markdown(str(roadmap["learning_outcome"]))
        weeks = _display_list(roadmap.get("weeks"))
        if not weeks:
            st.info("No roadmap items were generated.")
        for index, week in enumerate(weeks, start=1):
            week_data = _safe_dict(week)
            if not week_data:
                with st.expander(f"Week {index}", expanded=index == 1):
                    st.markdown(str(week))
                continue
            week_number = week_data.get("week") or index
            title = week_data.get("title") or f"Week {week_number}"
            with st.expander(f"Week {week_number}: {title}", expanded=index == 1):
                _render_list_section("Weekly Objectives", week_data.get("weekly_objectives"))
                _render_list_section("Activities", week_data.get("activities"))
                if week_data.get("deliverable"):
                    st.markdown("**Deliverable**")
                    st.markdown(str(week_data["deliverable"]))

    with lesson_tab:
        lesson = _safe_dict(media.get("lesson_plan"))
        st.markdown(f"**Skill:** {lesson.get('skill') or target_skill}")
        for label, key in [
            ("Learning Objectives", "learning_objectives"),
            ("Agenda", "agenda"),
            ("Activities", "activities"),
            ("Assessment", "assessment"),
        ]:
            with st.expander(label, expanded=key == "learning_objectives"):
                _render_list_items(lesson.get(key))

    with quiz_tab:
        quiz = _safe_dict(media.get("quiz"))
        mcqs = _display_list(quiz.get("mcqs"))
        short_answers = _display_list(quiz.get("short_answer_questions"))
        st.markdown("**Multiple Choice Questions**")
        if not mcqs:
            st.info("No multiple choice questions were generated.")
        for index, item in enumerate(mcqs, start=1):
            question = _safe_dict(item)
            title = _truncate(str(question.get("question") or f"Question {index}"), 90)
            with st.expander(f"MCQ {index}: {title}", expanded=index == 1):
                if question.get("question"):
                    st.markdown(str(question["question"]))
                _render_list_section("Options", question.get("options"))
                if question.get("answer"):
                    st.markdown("**Answer**")
                    st.markdown(str(question["answer"]))
                if question.get("explanation"):
                    st.markdown("**Explanation**")
                    st.markdown(str(question["explanation"]))

        st.markdown("**Short Answer Questions**")
        if not short_answers:
            st.info("No short answer questions were generated.")
        for index, item in enumerate(short_answers, start=1):
            question = _safe_dict(item)
            title = _truncate(str(question.get("question") or f"Question {index}"), 90)
            with st.expander(f"Short Answer {index}: {title}", expanded=index == 1):
                if question.get("question"):
                    st.markdown(str(question["question"]))
                if question.get("sample_answer"):
                    st.markdown("**Sample Answer**")
                    st.markdown(str(question["sample_answer"]))

    with assignment_tab:
        assignment = _safe_dict(media.get("practical_assignment"))
        st.markdown(f"**Project Title:** {assignment.get('project_title') or 'Practical Assignment'}")
        if assignment.get("problem_statement"):
            st.markdown("**Problem Statement**")
            st.markdown(str(assignment["problem_statement"]))
        for label, key in [("Deliverables", "deliverables"), ("Evaluation Criteria", "evaluation_criteria")]:
            with st.expander(label, expanded=True):
                _render_list_items(assignment.get(key))

    with project_tab:
        project = _safe_dict(media.get("mini_project"))
        st.markdown(f"**Project:** {project.get('title') or 'Mini Project'}")
        for label, key in [
            ("Architecture", "architecture"),
            ("Features", "features"),
            ("Tech Stack", "tech_stack"),
            ("GitHub Deliverables", "github_deliverables"),
        ]:
            with st.expander(label, expanded=key == "architecture"):
                _render_list_items(project.get(key))

    with presentation_tab:
        slides = _display_list(media.get("presentation_outline"))
        if not slides:
            st.info("No presentation outline was generated.")
        for index, item in enumerate(slides, start=1):
            slide = _safe_dict(item)
            slide_number = slide.get("slide") or index
            title = slide.get("title") or f"Slide {slide_number}"
            with st.expander(f"Slide {slide_number}: {title}", expanded=index == 1):
                _render_list_items(slide.get("bullet_points"))


def _render_list_section(label: str, values: Any) -> None:
    st.markdown(f"**{label}**")
    _render_list_items(values)


def _render_list_items(values: Any) -> None:
    items = _display_list(values)
    if not items:
        st.write("Not provided.")
        return
    for item in items:
        if isinstance(item, dict):
            parts = [f"{_format_key(key)}: {value}" for key, value in item.items() if value not in (None, "", [])]
            st.markdown(f"- {'; '.join(parts)}" if parts else "- Not provided.")
        else:
            st.markdown(f"- {item}")


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _display_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _format_key(value: str) -> str:
    return str(value).replace("_", " ").title()


def _truncate(value: str, limit: int) -> str:
    clean = value.strip()
    if len(clean) <= limit:
        return clean
    return f"{clean[: limit - 3].rstrip()}..."


def _ats_gauge(score: int) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0f8b8d"},
                "steps": [
                    {"range": [0, 50], "color": "#f6d2cd"},
                    {"range": [50, 75], "color": "#f8e7b6"},
                    {"range": [75, 100], "color": "#cce7df"},
                ],
            },
            title={"text": "ATS Score"},
        )
    )
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def _score_radar(report: ATSReport) -> go.Figure:
    categories = ["Skills", "Experience", "Education", "Certifications"]
    values = [
        round(report.score_breakdown.skills / 40 * 100),
        round(report.score_breakdown.experience / 30 * 100),
        round(report.score_breakdown.education / 20 * 100),
        round(report.score_breakdown.certifications / 10 * 100),
    ]
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=[*values, values[0]],
                theta=[*categories, categories[0]],
                fill="toself",
                name="Match Strength",
                line_color="#0f8b8d",
            )
        ]
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=340,
        margin=dict(l=40, r=40, t=40, b=30),
    )
    return fig


def _skill_pie(skill_analysis: SkillAnalysis) -> go.Figure:
    distribution = {
        key: value for key, value in skill_analysis.skill_distribution.items() if isinstance(value, int) and value > 0
    }
    if not distribution:
        distribution = {
            "Technical": len(skill_analysis.technical_skills),
            "Cloud": len(skill_analysis.cloud_skills),
            "DevOps": len(skill_analysis.devops_skills),
            "GenAI": len(skill_analysis.genai_skills),
            "Soft Skills": len(skill_analysis.soft_skills),
            "Tools": len(skill_analysis.tools),
        }
    frame = pd.DataFrame({"Category": list(distribution), "Count": list(distribution.values())})
    fig = px.pie(frame, values="Count", names="Category", title="Skill Distribution")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def _render_exports(
    resume: ResumeData | None,
    skill_analysis: SkillAnalysis | None,
    job_analysis: JobDescriptionAnalysis | None,
    ats_report: ATSReport,
    career_recommendation: CareerRecommendation | None,
    learning_media: dict[str, Any] | None = None,
) -> None:
    payload = {
        "resume": resume.model_dump() if resume else {},
        "skill_analysis": skill_analysis.model_dump() if skill_analysis else {},
        "job_description_analysis": job_analysis.model_dump() if job_analysis else {},
        "ats_report": ats_report.model_dump(),
        "career_recommendation": career_recommendation.model_dump() if career_recommendation else {},
        "learning_media": learning_media or {},
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    col_1, col_2, col_3 = st.columns(3)
    col_1.download_button(
        "Export JSON",
        data=json.dumps(payload, indent=2, ensure_ascii=False),
        file_name=f"ats_report_{timestamp}.json",
        mime="application/json",
        use_container_width=True,
    )
    col_2.download_button(
        "Export CSV",
        data=_build_csv_report(ats_report, career_recommendation),
        file_name=f"ats_report_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    col_3.download_button(
        "Download Report as PDF",
        data=_build_pdf_report(payload),
        file_name=f"ats_report_{timestamp}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def _build_csv_report(report: ATSReport, career_recommendation: CareerRecommendation | None) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["section", "item", "value"])
    writer.writeheader()
    writer.writerow({"section": "score", "item": "ATS Score", "value": report.ats_score})
    for key, value in report.score_breakdown.model_dump().items():
        writer.writerow({"section": "score_breakdown", "item": key, "value": value})
    for skill in report.matched_skills:
        writer.writerow({"section": "matched_skills", "item": skill, "value": ""})
    for skill in report.missing_skills:
        writer.writerow({"section": "missing_skills", "item": skill, "value": ""})
    for recommendation in report.recommendations:
        writer.writerow({"section": "recommendations", "item": recommendation, "value": ""})
    if career_recommendation:
        for role in career_recommendation.best_matching_roles:
            writer.writerow({"section": "career_paths", "item": role.role, "value": role.fit_score})
    return buffer.getvalue()


def _build_pdf_report(payload: dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter, title="ATS Resume Intelligence Report")
    styles = getSampleStyleSheet()
    story: list[Any] = []

    report = payload.get("ats_report", {})
    resume = payload.get("resume", {})
    career = payload.get("career_recommendation", {})

    story.append(Paragraph("ATS Resume Intelligence Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Candidate: {escape(resume.get('name') or 'Unknown')}", styles["Heading2"]))
    story.append(Paragraph(f"ATS Score: {report.get('ats_score', 0)}/100", styles["Heading2"]))
    story.append(Paragraph(escape(report.get("fit_summary") or ""), styles["BodyText"]))
    story.append(Spacer(1, 12))

    rows = [["Category", "Points"]]
    for key, value in (report.get("score_breakdown") or {}).items():
        rows.append([key.title(), str(value)])
    table = Table(rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f8b8d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))

    for title, key in [
        ("Matched Skills", "matched_skills"),
        ("Missing Skills", "missing_skills"),
        ("Recommendations", "recommendations"),
        ("Career Paths", "career_paths"),
    ]:
        values = report.get(key) or []
        if values:
            story.append(Paragraph(title, styles["Heading3"]))
            for value in values[:12]:
                story.append(Paragraph(f"- {escape(str(value))}", styles["BodyText"]))
            story.append(Spacer(1, 8))

    roles = career.get("best_matching_roles") or []
    if roles:
        story.append(Paragraph("Role Recommendations", styles["Heading3"]))
        for role in roles[:5]:
            text = f"{role.get('role', '')}: {role.get('fit_score', 0)}/100, {role.get('salary_band', '')}"
            story.append(Paragraph(escape(text), styles["BodyText"]))

    document.build(story)
    buffer.seek(0)
    return buffer.read()


def _add_history(resume: ResumeData, report: ATSReport, job_analysis: JobDescriptionAnalysis) -> None:
    st.session_state.history.insert(
        0,
        {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Candidate": resume.name or resume.source_file or "Unknown",
            "Role": job_analysis.role_title or "Unknown",
            "ATS Score": report.ats_score,
        },
    )
    st.session_state.history = st.session_state.history[:10]


def _render_history() -> None:
    history = st.session_state.get("history") or []
    if not history:
        return
    with st.expander("Session History"):
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


def _dict_to_frame(values: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame([{"Field": key, "Value": value} for key, value in values.items() if value])


if __name__ == "__main__":
    main()
