# ATS Resume Intelligence Agent

A production-ready Streamlit application that demonstrates Azure AI Foundry Content Understanding and Azure OpenAI agentic analysis for resume parsing, ATS matching, skill gap analysis, and career recommendations.

## Features

- Resume upload for PDF and DOCX files
- Azure AI Content Understanding document extraction
- Azure OpenAI GPT-4.1 or GPT-4o powered resume parsing
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
|   `-- jd_matcher.py
|-- models/
|   `-- ats_report.py
`-- utils/
    `-- prompts.py
```

## Azure Configuration

Create or use existing Azure resources:

1. Azure OpenAI resource with a GPT-4.1 or GPT-4o deployment.
2. Microsoft Foundry or Azure AI Services resource that supports Azure AI Content Understanding.
3. A Content Understanding analyzer. The default is `prebuilt-documentSearch`; replace it with your custom deployed analyzer ID if you already created one for resumes.

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

Update `.env`:

```env
AZURE_OPENAI_ENDPOINT=https://<your-openai-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-azure-openai-api-key>
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
AZURE_OPENAI_API_VERSION=2025-01-01-preview

CONTENTUNDERSTANDING_ENDPOINT=https://<your-foundry-resource>.services.ai.azure.com/
CONTENTUNDERSTANDING_KEY=<optional-content-understanding-key>
CONTENTUNDERSTANDING_ANALYZER_ID=prebuilt-documentSearch
CONTENTUNDERSTANDING_API_VERSION=2025-11-01
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

The sidebar lets you override all Azure values at runtime without editing files.

## Workflow

1. Upload a PDF or DOCX resume.
2. Click `Extract Resume`.
3. Paste a job description.
4. Click `Analyze ATS Score`.
5. Click `Generate Recommendations` to refresh career guidance.
6. Export the report as JSON, CSV, or PDF.

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

- Azure Content Understanding is used first for document extraction.
- Local PDF/DOCX parsing is included as a development fallback.
- Azure OpenAI is required for structured resume parsing, skill analysis, JD analysis, matching, and recommendations.
- The app enforces the ATS score formula in Python after the model returns its analysis, so scoring stays consistent.

## References

- Azure AI Content Understanding Python SDK: https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/contentunderstanding/azure-ai-contentunderstanding
- Azure AI Content Understanding documentation: https://learn.microsoft.com/azure/ai-services/content-understanding/
