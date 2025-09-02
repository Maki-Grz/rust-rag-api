# Rust RAG API

**Internal Rust API for retrieval-augmented generation using document embeddings.**

## Overview

This project implements an internal API in Rust using Actix-web for Retrieval-Augmented Generation (RAG).  
The API allows you to ingest documentation, store it in a vector database (e.g., Azure Cosmos DB), and query it with natural language questions.  
All responses are based strictly on the ingested documentation.

## Features

- Ingest documents and segment them into passages
- Generate embeddings for each passage and store in a vector database
- Retrieve top-k relevant passages for a given question
- Generate answers based on retrieved passages using a language model
- Internal API endpoints:
    - `POST /ingest` – add new documents
    - `POST /ask` – ask a question and receive an answer

## Architecture

- **ingestion**: text segmentation, embedding creation, storage
- **retrieval**: compute question embedding, search top-k passages
- **generation**: produce answer from question + retrieved passages
- **api**: Actix-web endpoints to handle requests

## Setup

1. Clone the repository
2. Configure Cosmos DB credentials
3. Build the project using Cargo
4. Run the API

## Environment Variables

````dotenv
COSMOS_URI=mongodb://
DATABASE=db-name
COLLECTION=collection-name
LLM_URI=http://localhost:12434/engines/llama.cpp/v1/chat/completions
````

LLM_URI is the URL of the language model with Docker Models.

## Future Improvements

- Support for updating or deleting passages (Coming soon)
- Optimized passage selection for long documents (In progress)
- Integration with more powerful language models (in progress)
- Optimized queries for large collections (In progress)

Note: This project is still in development.
Some functionnality (docs, structure) may be developed with the help of AI-assisted development tools, but the majority of the code is written by hand.

Commit messages are generated with JetBrains AI Assistant.