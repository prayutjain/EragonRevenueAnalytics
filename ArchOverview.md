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
│  Frontend    │  <=>  │ FastAPI Server │  <=>  │   Analytics Engine  │
│  (React)     │       │ (LLM Orchestration)    │ (Pandas + OpenAI + PDF)    │
└──────────────┘       └────────────────┘       └─────────────────────┘
                            │
                            ▼
                      ┌─────────────┐
                      │   Database  │ (CSV or PostgreSQL)
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

## 🧠 3. LLM Orchestration

### Purpose:

To handle natural language inputs and convert them into structured, explainable, and actionable insights using OpenAI models and function calling.

### Flow:

1. **Receive Query:**

   * Endpoint `/api/query/stream` receives a user query + optional `conversation_id`
2. **Step-by-Step LLM Pipeline:**

   * **Step 1: Query Classification**
     Identifies type (e.g., ranking, aggregation, visualization request)
   * **Step 2: Function Planning**
     LLM suggests one or more function calls (e.g., `get_top_accounts`, `analyze_win_rate`)
   * **Step 3: Function Execution**
     Backend executes Pandas-based functions and returns JSON
   * **Step 4: Insight Generation**
     LLM generates natural language explanation of results
   * **Step 5: Visualization Planning**
     LLM selects chart type(s) and assigns data keys and formatting
   * **Step 6: Final Assembly**
     Returns: `answer`, `key_metrics`, `visualizations`, `functions_executed`, `data_sources`

### Highlights:

* ✅ OpenAI function calling for grounded reasoning
* ✅ Supports multi-turn conversations via `conversation_id`
* ✅ Structured LLM output used directly for frontend rendering
* ✅ Streaming support for real-time feedback

### Example Function Schema:

```json
{
  "function": "get_top_accounts_by_value",
  "args": {
    "limit": 5,
    "min_amount": 10000
  },
  "response": [
    {
      "account_name": "Acme Corp",
      "total_value": 1250000,
      "deal_count": 4
    }
  ]
}
```

---

## ⚙️ 4. Deployment Overview

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

## 🔐 5. Security Considerations

* ✅ OpenAI key never exposed to frontend
* ✅ API rate-limiting can be enforced with middleware
* ✅ CORS enabled selectively for frontend domain
* ✅ PDF generation is sandboxed from user uploads

---

## 📈 6. Future Enhancements

* [ ] PostgreSQL-based data pipeline with upload UI
* [ ] Admin panel for dataset management
* [ ] User authentication for multi-user support
* [ ] Interactive Dashboard with saved views
* [ ] GPT-4o multimodal support (CSV uploads, charts)

