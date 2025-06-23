## 🧭 Revenue Analytics System – Architecture Overview

### 📌 Objective

To build an AI-powered analytics platform that enables CROs and revenue teams to:

* Ask natural language questions
* View dynamic visualizations & reports
* Understand product and sales performance across geography and segments

---

## 🏗️ 1. High-Level Architecture

```
┌──────────────┐       ┌────────────────┐       ┌─────────────────────┐
│  Frontend         │  <=>  │ FastAPI Server      │  <=>  │   Analytics Engine         │
│  (React)          │       │ (LLM Orchestration)         │ (Pandas + OpenAI + PDF)    │
└──────────────┘       └────────────────┘       └─────────────────────┘
                            │
                            ▼
                      ┌─────────────┐
                      │   Database      │ (CSV or PostgreSQL)
                      └─────────────┘
```

---

## 🧩 2. Component Breakdown

### 🔹 Frontend (React + Recharts)

* **UI Library:** Lucide, Recharts
* **Key Components:**

  * `RevenueAnalyticsDashboard`: Main container
  * `SmartChart`: Renders charts (bar, pie, line, table)
  * `StreamingStatus`, `PDFDownloadButton`, `SystemStatusIndicator`
* **Features:**

  * Natural language chat with AI
  * Quick questions panel
  * Dynamic data visualizations
  * PDF export of insights
  * Responsive design (mobile + desktop)

---

### 🔹 Backend (FastAPI)

* **Modules:**

  * `query_stream`: Receives NL questions, returns LLM results via SSE
  * `pdf_report`: Generates styled PDF reports (ReportLab)
  * `data_loader`: Loads and caches opportunity/contact data
  * `health`: Reports system status (data loaded, counts)
* **LLM Integration:**

  * Uses OpenAI GPT models for summarization and insight generation
  * Performs multiple function calls:

    * Query classification
    * Function selection
    * Answer & visualization generation
* **PDF Output:**

  * Key Metrics + Visualizations + Data Source Citations
  * Supports bar, pie, line, and table charts

---

### 🔹 Analytics Engine

* **Tech Stack:** Pandas, NumPy, ReportLab, Recharts-ready JSON
* **Capabilities:**

  * Aggregation & grouping (e.g., top accounts by value)
  * Win rate, stage-wise funnel analysis
  * Role frequency, title analysis, product conversion rates
  * Forecasting pipeline revenue from probabilities

---

### 🔹 Data Layer

* **Current:** CSV file ingestion (opportunities, contacts)
* **Planned:** PostgreSQL for scalability
* **Schema:**

  * `opportunities`: account, stage, amount, probability, etc.
  * `contacts`: name, title, opportunity\_id

---

## ⚙️ 3. Deployment Overview

### Dev Environment

* **Frontend:** React
* **Backend:** FastAPI run via `uvicorn`
* **API Base URL:** `http://localhost:8083`

### Production Options

* **Backend:** Deploy to Render, Railway, or EC2
* **Frontend:** Vercel / Netlify (builds from `src/`)
* **Env Setup:**

  * Frontend: Use proxy or call backend only (no key exposed)

---

## 🔐 4. Security Considerations

* ✅ OpenAI key never exposed to frontend
* ✅ API rate-limiting can be enforced with middleware
* ✅ CORS enabled selectively for frontend domain
* ✅ PDF generation is sandboxed from user uploads

---

## 📈 5. Future Enhancements

* [ ] PostgreSQL-based data pipeline with upload UI
* [ ] Admin panel for dataset management
* [ ] User authentication for multi-user support
* [ ] Interactive Dashboard with saved views
* [ ] GPT-4o multimodal support (CSV uploads, charts)

