Autonomous Product Discovery Agent (RAG)

Project Overview
This project is an AI-powered product discovery system that combines Retrieval-Augmented Generation (RAG) with the ReAct (Reasoning and Acting) framework. Unlike traditional keyword-based search, this agent understands user intent (e.g., "something for a daily commute") and autonomously queries a semantic vector store to provide grounded, relevant recommendations.

Purpose & Impact
Semantic Intelligence: Moves beyond keyword matching to capture user context and product benefits.

Grounded Reasoning: Uses a search tool to ensure recommendations are based on actual inventory, significantly reducing LLM hallucinations.

Autonomous Workflows: Implements a ReAct loop where the model "thinks" about the query before deciding which search tool to invoke.

The Tech Stack
Orchestration: LangChain v1.0 (Modern graph-based agent architecture)

LLM: Lastest Google Gemini Flash (Optimized for speed and high-frequency tool use)

Vector Database: FAISS (High-speed, in-memory semantic retrieval)

Embedding Model: all-MiniLM-L6-v2 via HuggingFace (Context-aware text-to-vector conversion)

Environment: Python 3.10+ and python-dotenv for secure credential management

Getting Started
1. Prerequisites
Python 3.10 or higher

A Google Gemini API Key (Get one for free at (https://aistudio.google.com/))

2. Installation

Clone the repository and install the required dependencies:

Create and activate a virtual environment

Bash

python -m venv venv

source venv/bin/activate  # Windows:.\venv\Scripts\activate

Install libraries
pip install langchain langchain-google-genai langchain-huggingface langchain-community faiss-cpu python-dotenv

3. Security Configuration
Create a .env file in the root directory and add your API key:

Code snippet

GOOGLE_API_KEY=your_gemini_api_key_here

4. Running the Agent
Execute the main script to start a session with the discovery agent:

Bash

python main.py

How It Works: The ReAct Loop
The agent follows a cyclic reasoning pattern for every user query:

Thought: The model analyzes the input (e.g., "I need a quiet workspace")

Action: It decides to call the search_products tool

Observation: It retrieves items like "Noise-Canceling Headphones" from the FAISS vector store

Final Response: It synthesizes the retrieved data into a natural language recommendation

Future Enhancements
Multi-Step Reasoning: Integrating a "Price Filter" tool to handle budget constraints via SQL

Observability: Adding LangSmith tracing to visualize the agent's internal thought process

Persistence: Migrating to LangGraph for long-term user memory across sessions