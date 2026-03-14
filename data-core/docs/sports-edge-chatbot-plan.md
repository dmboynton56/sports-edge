# Plan: Sports-Edge Chatbot & Knowledge Base

## Goal
Empower the chatbot on the Sports-Edge deep dive page to accurately and comprehensively answer user questions about the project's architecture, methodologies, models, and the historical data collected for generating predictions.

## Phase 1: Knowledge Base Aggregation
1. **Document the Architecture & Methodology:** Compile all technical documentation, model architectures, feature engineering choices, and the rationale behind your predictive models into markdown files.
2. **Data Dictionaries:** Create detailed summaries of the historical data collected (e.g., years covered, types of bets, variables like weather/injuries, and data sources).
3. **Project Narrative:** Write out the story of the project, including challenges faced, why certain frameworks were chosen, and the ultimate vision. This gives the bot a "personality" and deep context.

## Phase 2: Vector Database & RAG Implementation
1. **Chunking and Embedding:** Process the aggregated markdown and documentation files, split them into logical chunks, and generate embeddings using an embedding model (like OpenAI's `text-embedding-3-small`).
2. **Vector Store Setup:** Store these embeddings in a vector database (e.g., Supabase pgvector, Pinecone, or Qdrant).
3. **Retrieval-Augmented Generation (RAG):** When a user asks a question, embed the query, retrieve the top `K` most relevant chunks from the vector database, and inject them into the LLM's system prompt to ground its answer.

## Phase 3: Advanced Data Querying (Optional but Powerful)
1. **Text-to-SQL / API Tooling:** If users want to ask specific statistical questions ("What was the model's win rate on NFL underdogs last season?"), standard RAG on text documents might fail.
2. **Function Calling:** Give the chatbot access to a function (tool) that can query your PostgreSQL database directly, or hit your Sports Edge API to retrieve live/historical stats on the fly.

## Phase 4: UI & Chatbot Integration
1. **Context-Aware Prompting:** Design a strong system prompt that defines the chatbot's role as the "Sports-Edge Project Expert." Instruct it to rely strictly on provided context and to admit when it doesn't know an answer.
2. **Chat Interface:** Embed the chat component directly on the deep dive page (using Vercel AI SDK or similar for streaming responses).
3. **Suggested Questions:** Pre-populate the chat UI with 3-4 suggested questions (e.g., "How does the Daily Bias Prediction work?", "What data do you use?") to guide user interaction immediately.
