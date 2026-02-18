# NINA - A Lightweight Multi-Mode LLM System for Low Resource Deployment

NINA is a modular, offline-first AI system designed to operate efficiently on resource-constrained hardware while supporting multiple practical user workflows. The project integrates small local language models with classical computing tools and external data sources to provide a cohesive, multi-function assistant suitable for desktop applications and future web-based interfaces. The system is built around a flexible mode architecture, allowing each capability to remain isolated, maintainable, and independently extendable.
<p align="center">  <img src="static/Images/Image1.jpg" width="70%"> </p>

---
## Core Capabilities
NINA provides four primary operational modes:

### 1️. General Assistant
A lightweight instruction-tuned model (**Llama-3.2-1B-Instruct** or **Qwen-1.5-0.5B-Chat**) used for general queries, reasoning and conversational utilities.

**Features**
- Local inference (CPU/GPU auto-detection)
- Controlled context growth with history trimming
- VRAM-aware prompt caps
- Deterministic operation via configurable seeds and parameters
- Safe generation safeguards

### 2️. Math Solver
A hybrid computation module combining model-driven reasoning with symbolic mathematics.

**Features**
- Arithmetic and algebraic evaluation
- Geometry and measurement problems
- Basic statistics
- Word problem interpretation
- Safe symbolic computation via SymPy
- Clear separation between reasoning and exact calculation

### 3️. Translator
A structured translation module supporting more than 20 languages.

**Features**
- Strict translation behavior enforced through system prompting
- Controlled output (no explanatory augmentation)
- Language pair selection and mid-conversation reconfiguration
- Input length controls to maintain performance stability

### 4️. Local Guide
A location-aware information retrieval module integrating external data APIs and offline datasets.

**Capabilities include**
- Weather information (OpenWeatherMap)
- News summaries (GNews)
- Tourist locations, supermarkets, and hotels (Geoapify Places)
- Flight information (AviationStack)
- Automatic city detection from user input
- Intent classification for routing requests
- Local airport code resolution via offline airport dataset

**Example Queries**
- “I want to visit places in Berlin.”
- “Where can I buy groceries in Frankfurt?”
- “Find hotels in Munich.”
- “Show flights from FRA to JFK.”

---


## Technology Stack
### 1. Language Models
* **Llama-3.2-1B-Instruct** (primary model across all modes)
* **Qwen-1.5-0.5B-Chat** (optional alternative)
* Both loaded in **4-bit quantization** to operate within 4 GB VRAM.

### 2. Programming Environment
* Python 3.10+
* Libraries:
   * torch
   * transformers
   * bitsandbytes
   * huggingface_hub
   * flask
   * python-dotenv
   * requests
   * psutil

### 3 External Data Sources
* **OpenWeatherMap** – weather forecasts
* **GNews** – news headlines
* **Geoapify Places API** – tourist attractions, supermarkets, hotels
* **AviationStack API** – flight data
* **OpenFlights Airport Dataset** – IATA code resolution

---


## 🏗 System Architecture

