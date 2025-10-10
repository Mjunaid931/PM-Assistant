# Product Requirements Document: RAG PM Assistant (v1.0 MVP)

## 1. Goal & Strategic Objective
To reduce the time spent by internal teams (Sales, Engineering) searching through static product documentation, thereby improving sales velocity and decreasing technical support burden.

**Success Metric (KPI):** Reduce the average time to retrieve a key technical spec or pricing detail from 5 minutes (manual search) to under 30 seconds (via RAG assistant).

## 2. Target User Persona
* **Persona:** Sarah, a Mid-Level Sales Representative
* **Pain Point:** Frequently needs to confirm API rate limits or overage pricing while on a call, but documentation is fragmented across various PDFs and text files.

## 3. Product Features (Scope)

| Feature ID | Feature | Priority | RAG Component Demonstrated |
| :--- | :--- | :--- | :--- |
| P-01 | **Context-Aware Q&A** | High | Retrieval (Vector Search) and Generation (LLM) |
| P-02 | **Context Grounding** | High | Optimized Prompt to prevent hallucination |
| P-03 | **Simple Web Interface** | Medium | Streamlit UI (Accessibility) |
| P-04 | **Source Attribution** | Low (Future) | Metadata filtering on documents |

## 4. Technical Specifications & Constraints
* **LLM:** OpenAI GPT-4o-mini (chosen for speed and cost-efficiency).
* **Embedding Model:** OpenAI `text-embedding-3-small` (chosen for performance).
* **Vector Store:** ChromaDB (chosen for quick, local prototyping).
* **Data Sources:** Initial MVP is limited to `.txt` files in the `/knowledge_base` directory.

## 5. Iteration & Future Roadmap
* **V1.1:** Add PDF document support (`pypdf` loader).
* **V1.2:** Implement Multi-turn Chat Memory (for conversational history).
* **V2.0:** Introduce **Post-Retrieval Filtering** to improve context quality.