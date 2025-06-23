# 🧠 Revenue Analytics Backend

**Version:** 1.0.0
**Purpose:** An AI-powered analytics system helping CROs understand enterprise sales performance, identify bottlenecks, and generate insights across geographies and product lines.

---

## 🚀 Key Features

* 🔍 **Natural Language Understanding**: Ask questions like *"What’s the pipeline value by stage?"*
* 📊 **LLM-Powered Visualizations**: Automatically generate bar, pie, line, or table visuals from query results
* 📈 **Executive KPI Summaries**: Extract top metrics like win rate, average deal size, pipeline value
* 📄 **Automated PDF Reports**: Generate CRO-ready reports with charts, tables, and insights
* 🔁 **Streaming Insights**: Real-time, multi-turn responses for follow-up queries

---

## 🛠️ Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/prayutjain/revenue-analytics.git
   python -m venv .venv
   .venv/bin/activate
   cd revenue-analytics-backend
   ```

2. **Install Dependencies**
   Make sure you’re using Python 3.9+ and install requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set OpenAI API Key**

   ```bash
   export OPENAI_API_KEY='your-openai-key-here'
   ```

4. **Place Required CSVs in Project Root**

   * `opportunities.csv`
   * `account_and_contact.csv`

5. **Run the Server**

   ```bash
   python enhanced_analytics_server.py
   ```

--- 

## 🧱 Architecture Overview

```plaintext
[ FastAPI App ]
     |
     ├── /api/query           → LLM-powered insights and summarization
     ├── /api/query/stream    → Real-time query responses with context
     ├── /api/report/pdf      → Generate PDF from conversation
     ├── /api/conversations   → List of saved conversations
     ├── /api/data/schema     → View loaded schema and columns
     └── /api/quick-insights  → Pre-baked insights (metrics and counts)
```

* **LLM Calls**: Uses GPT-4o to process queries, determine functions, and format results
* **Function Registry**: Modular function handlers (e.g., `get_top_accounts_by_total_value`)
* **Conversation Memory**: Each query/response saved with context for PDF generation
* **Visualization Engine**: Uses `matplotlib` + `reportlab` to render charts and tables

---

## 🔌 API Documentation

### `POST /api/query/stream`

Ask questions like:

```json
{
  "question": "What are the top 5 accounts by opportunity value?"
}
```

**Returns:**

* `answer`: Executive summary (from LLM)
* `visualizations`: Bar, pie, line, or table chart data
* `key_metrics`: High-level KPI cards
* `functions_executed`: List of analytics functions run

---

### `POST /api/report/pdf`

Generate PDF report:

```json
{
  "conversation_id": "conv_20250623_101230",
  "title": "Q2 2025 Revenue Report"
}
```

Returns a downloadable PDF with:

* Query
* Summary
* KPIs
* Charts
* Tables
* Key insights

---

## 🧪 Sample Questions & Outputs

1. **"What are the top 5 accounts by opportunity value?"**

   * ✅ Executed: `get_top_accounts_by_total_value`
   * 📊 Visualization: Bar chart
   * 💡 Insights: Contribution by account, deal count, win/loss mix

2. **"Compare win rates by product category"**

   * ✅ Executed: `analyze_product_win_rates`
   * 📈 Visualization: Bar + table
   * 💡 Insights: Highest converting products, potential investment focus

3. **"How is our pipeline trending over time?"**

   * ✅ Executed: `calculate_pipeline_trends`
   * 📉 Visualization: Line chart
   * 💡 Insights: Decline or growth patterns, volatility

---

## 📂 Data Requirements

* **`opportunities.csv`**

  * Columns: `Account Name`, `Opportunity Name`, `Stage`, `Amount`, `Probability (%)`, `Close Date`, `Created Date`, etc.

* **`account_and_contact.csv`**

  * Columns: `Account Name`, `Title`, `Primary Contact`, etc.

---

## 📎 Contributing

This backend is designed for extensibility. Add new functions to `FUNCTION_HANDLERS` and update the `/api/query` route to integrate them seamlessly.

---

## 🖥️ Frontend Setup (React UI)

The frontend is a highly interactive React app that connects to the FastAPI backend to render visual analytics, KPIs, streaming responses, and downloadable PDF reports.

### 📦 Prerequisites

Ensure you have the following installed:

* **Node.js v18+**
* **npm** or **yarn**
* Backend running locally at `http://localhost:8083`

---

### 📁 Folder Structure

Place your frontend code under a directory like:

```
project-root/
├── backend/
│   └── enhanced_analytics_server.py
├── frontend/
│   ├── public/
│   ├── src/
│   ├── styles/
│   │   └── Dashboard.css
│   │   └── Visualizations.css
│   │   └── Animations.css
│   │   └── Typography.css
│   │   └── Responsive.css
│   ├── App.jsx
│   └── index.js
├── package.json
```

---

### 🚀 Install & Run Frontend

```bash
cd frontend
npm install
npm run dev
```

This will run your frontend app at:

```
http://localhost:3002
```

> Make sure the `API_BASE_URL` in your component (currently set to `http://localhost:8083`) matches your backend URL.

---

### 🔧 Recommended Setup for `App.jsx`

```jsx
import React from 'react';
import RevenueAnalyticsDashboard from './components/RevenueAnalyticsDashboard';

function App() {
  return (
    <div className="App">
      <RevenueAnalyticsDashboard />
    </div>
  );
}

export default App;
```

### 🌐 CORS Support

Ensure your FastAPI backend has CORS middleware enabled:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3002"],  # or "*" for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

