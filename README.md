## Project Overview

This repository is based on the **LangChain** framework and features a custom **MoA (Mixture of Agents)** model. It utilizes **RAG (Retrieval-Augmented Generation)** to enable question-answering from a local knowledge base.

## What is MoA?

MoA (Mixture of Agents) is a novel approach that mixes multiple Large Language Model (LLM) agents to handle tasks without the need to adjust LLM parameters. Unlike the traditional MoE (Mixture of Experts) method, which typically involves parameter adjustments of expert models, MoA combines outputs from multiple models and feeds these outputs into an aggregation LLM to generate the final result. This approach allows for more flexible and efficient task handling. MoA supports various models, such as **LLaMA 3.1, GPT-4**, and others. You can reference the model zoo(https://docs.together.ai/docs/chat-models) for available models. If a model is not present, you will need to deploy it yourself outside of the together library. Future updates will include improvements to this section of the code.

## Usage Guide

### Environment Setup

First, ensure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
```
### Creating the Knowledge Base

Before you can start asking questions, you need to create the knowledge base. Run the following script to populate the local knowledge base:

```bash
python populate_database.py
```
### Asking Questions

Once the knowledge base is created, you can use the following script to ask questions and get answers:

```bash
python ask_question_with_rag.py
```

## File Descriptions

- `populate_database.py`: Script to create and populate the local knowledge base.
- `ask_question_with_rag.py`: Script to ask questions to the local knowledge base and retrieve answers.

## References

This project is inspired by and builds upon the work from the following repositories:
- [LangChain](GitHub - langchain-ai/langchain: 🦜🔗 Build context-aware reasoning applications)
- [MoA](https://github.com/togethercomputer/MoA)
- [RAG](https://github.com/pixegami/rag-tutorial-v2)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to improve the project. If you have any questions or suggestions, you can reach out by opening an issue.

