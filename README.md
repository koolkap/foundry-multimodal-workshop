# ATS Resume Intelligence Agent

A production-ready Streamlit application that demonstrates Azure AI Foundry Content Understanding for resume parsing, ATS matching, skill gap analysis, and career recommendations.

## Features

- Resume upload for PDF and DOCX files
- Azure AI Content Understanding document extraction with a CU Studio analyzer
- Local deterministic ATS skill analysis, job matching, scoring, and recommendations
- Structured Pydantic JSON outputs
- Job description analysis
- ATS score from 0 to 100 using:
  - 40% skills
  - 30% experience
  - 20% education
  - 10% certifications
- Plotly dashboard:
  - ATS score gauge
  - Skill match radar chart
  - Skill distribution pie chart
  - Missing skills table
  - Recommendations table
- Career recommendations:
  - Best matching roles
  - Salary bands
  - Learning path
  - Missing technologies
- Learning Media Generator powered by your configured Azure OpenAI deployment:
  - 8 week learning roadmap
  - Lesson plan
  - Quiz
  - Practical assignment
  - Mini project
  - Presentation outline
- Export JSON, CSV, and PDF reports
- Session history
- Dark mode toggle

## Project Structure

```text
.
|-- app.py
|-- main.py
|-- requirements.txt
|-- .env.example
|-- services/
|   |-- document_processor.py
|   |-- ats_agent.py
|   |-- learning_media.py
|   `-- jd_matcher.py
|-- models/
|   `-- ats_report.py
`-- utils/
    `-- prompts.py
```

## Azure Configuration

Create or use existing Azure resources:

1. Microsoft Foundry or Azure AI Services resource that supports Azure AI Content Understanding.
2. A Content Understanding analyzer created in CU Studio. The default analyzer ID in this app is `ats`.

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

Update `.env`:

```env
CONTENTUNDERSTANDING_ENDPOINT=https://<your-foundry-resource>.services.ai.azure.com/
CONTENTUNDERSTANDING_KEY=<optional-content-understanding-key>
CONTENTUNDERSTANDING_ANALYZER_ID=ats
CONTENTUNDERSTANDING_API_VERSION=2025-11-01
AZURE_OPENAI_ENDPOINT=https://<your-azure-openai-resource>.openai.azure.com/
# Foundry/Azure AI Services deployments may instead use:
# AZURE_OPENAI_ENDPOINT=https://<your-foundry-resource>.cognitiveservices.azure.com/
AZURE_OPENAI_API_KEY=<your-azure-openai-key>
AZURE_OPENAI_DEPLOYMENT=<your-model-deployment-name>
AZURE_OPENAI_API_VERSION=<your-azure-openai-api-version>
```

If `CONTENTUNDERSTANDING_KEY` is empty, the app uses `DefaultAzureCredential`. In that case, sign in locally with Azure CLI and ensure the signed-in identity has the required Cognitive Services permissions.

## Installation

Use Python 3.11 or later.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

The sidebar lets you override Content Understanding endpoint, analyzer ID, API version, and key behavior at runtime without editing files.

## Workflow

1. Upload a PDF or DOCX resume.
2. Click `Extract Resume`.
3. Paste a job description.
4. Click `Analyze ATS Score`.
5. Click `Generate Recommendations` to refresh career guidance.
6. Click `Generate Learning Media` to create a personalized roadmap, lesson plan, quiz, assignment, mini project, and presentation outline.
7. Export the report as JSON, CSV, or PDF.

## Output Example

```json
{
  "ats_score": 85,
  "matched_skills": [],
  "missing_skills": [],
  "strong_skills": [],
  "recommendations": [],
  "career_paths": []
}
```

## Notes

- Azure Content Understanding is used first for document extraction and structured analyzer output.
- Local PDF/DOCX parsing is included as a development fallback.
- ATS scoring and matching stay deterministic in Python.
- Azure OpenAI is used only for the Learning Media Generator and reads the existing `AZURE_OPENAI_*` settings from the environment.

## References

- Azure AI Content Understanding Python SDK: https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/contentunderstanding/azure-ai-contentunderstanding
- Azure AI Content Understanding documentation: https://learn.microsoft.com/azure/ai-services/content-understanding/
