## Project Overview

This repository is based on the **LangChain** framework and features a custom **MoA (Mixture of Agents)** model. It utilizes **RAG (Retrieval-Augmented Generation)** to enable question-answering from a local knowledge base.

## What is MoA?

MoA (Mixture of Agents) is a novel approach that mixes multiple Large Language Model (LLM) agents to handle tasks without the need to adjust LLM parameters. Unlike the traditional MoE (Mixture of Experts) method, which typically involves parameter adjustments of expert models, MoA combines outputs from multiple models and feeds these outputs into an aggregation LLM to generate the final result. This approach allows for more flexible and efficient task handling.

## Usage Guide

### Environment Setup

First, ensure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
