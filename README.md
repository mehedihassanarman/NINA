# NINA
_A Lightweight Multi-Mode LLM System for Low Resource Deployment_
---

NINA is a modular, offline-first AI system designed to operate efficiently on resource-constrained hardware while supporting multiple practical user workflows. The project integrates small local language models with classical computing tools and external data sources to provide a cohesive, multi-function assistant suitable for desktop applications and future web-based interfaces.

The system is built around a flexible mode architecture, allowing each capability to remain isolated, maintainable, and independently extendable.

---

## 🚀 Core Capabilities

NINA provides four primary operational modes:

---

### 1️⃣ General Assistant

A lightweight instruction-tuned model (**Llama-3.2-1B-Instruct**) used for general queries, reasoning, and conversational utilities.

**Features**
- Local inference (CPU/GPU auto-detection)
- Controlled context growth with history trimming
- VRAM-aware prompt caps
- Deterministic operation via configurable seeds and parameters
- Safe generation safeguards

---

### 2️⃣ Math Solver

A hybrid computation module combining model-driven reasoning with symbolic mathematics.

**Features**
- Arithmetic and algebraic evaluation
- Geometry and measurement problems
- Basic statistics
- Word problem interpretation
- Safe symbolic computation via SymPy
- Clear separation between reasoning and exact calculation

---

### 3️⃣ Translator

A structured translation module supporting more than 20 languages.

**Features**
- Strict translation behavior enforced through system prompting
- Controlled output (no explanatory augmentation)
- Language pair selection and mid-conversation reconfiguration
- Input length controls to maintain performance stability

---

### 4️⃣ Local Guide

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

## 🏗 System Architecture

