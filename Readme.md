# Freight Tiger - Smart Logistics Assistant

## Approach
This project processes shipping data to identify unjustified cost increases using mathematical baselines and a Retrieval-Augmented Generation (RAG) AI layer.

### 1. Core Logic & Math
- **Cost Calculation:** Aggregated shipments by `route`, `route_type`, and the Monday of `week_of` to find the average cost per tonne-km.
- **Historical Baseline:** Implemented a strict 8-week rolling average. By utilizing Pandas `shift(1)`, the system guarantees zero look-ahead bias (it only uses history).
- **Peer Baseline:** Calculated the average cost of all routes with the same `route_type` in the same week, explicitly subtracting the current route's total from the peer aggregate to exclude it from its own baseline.

### 2. AI & RAG Implementation
If a route's cost rises more than 10% above both baselines, the script triggers an AI evaluation. 
- **Retrieval:** It filters the `context_notes.csv` file for events that apply to either the specific route or 'All Routes' up to that date.
- **Evaluation:** Using the Groq API (openai/gpt-oss-20b), the LLM strictly evaluates if the retrieved notes justify the cost spike. 
- **Hallucination Prevention:** The prompt forces a strict JSON schema. If no note matches, it explicitly flags the route as unexplained.

### 3. Reproducibility
The LLM temperature was strictly set to `0.0`. I ran the script three consecutive times on the same dataset, and a diff check confirmed the output CSVs were 100% identical across all runs.

### 4. Cost & Token Log
- **Model Used:** `llama3-70b-8192` (via Groq Free Tier API)
- **Total LLM Calls:** 7 API calls (only spiked routes were queried to save resources).
- **Cost:** $0.00 (Groq free tier).