# Development with Langchain

Welcome to the Langchain Getting Started Guide! This guide will help you understand the core concepts of Langchain and provide examples to get you started with developing applications using large language models (LLMs).

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Connecting to an LLM Model](#connecting-to-an-llm-model)
  - [OpenAI](#openai)
  - [Local LLM](#local-llm)
- [Prompts](#prompts)
  - [Basic Prompts](#basic-prompts)
  - [Prompts with Structured Output](#prompts-with-structured-output)
- [Embeddings](#embeddings)
- [Indexes / Vector Databases](#indexes--vector-databases)
- [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
- [Agents and Tools](#agents-and-tools)

## Introduction

Langchain is an open-source orchestration framework for the development of applications that use large language models. It provides a standard interface for interacting with many different LLMs.

## Installation

To get started with Langchain, install the necessary packages using pip:

```bash
%pip install langchain langchain-core langchain-community langchain-openai langchain-chroma langchainhub tavily-python sqlalchemy==1.4.46 snowflake-sqlalchemy snowflake-connector-python
```

## Connecting to an LLM Model
# OpenAI
To connect to an OpenAI LLM, use the following code snippet:

```python
from langchain_openai import OpenAI

# Pass in your OpenAI API key as an environment variable OPENAI_API_KEY
openai_model = OpenAI()
result = openai_model.invoke("what are the SOLID principles in programming?")
print(result)
```
# Local LLM
For local LLM, download and install Ollama. After installation, run ollama run mistral to expose the model on http://localhost:11434.
```python
from langchain_community.llms import Ollama

mistral_model = Ollama(model="mistral")
llama3_model = Ollama(model="llama3")
result = mistral_model.invoke("what are the SOLID principles in programming?")
print(result)
result = llama3_model.invoke("what are the SOLID principles in programming?")
print(result)
```
## Prompts
# Basic Prompts
Prompts are a set of instructions given to a large language model. The PromptTemplate class in Langchain formalizes the composition of prompts without the need to manually hardcode context and query.
```python
from langchain_core.prompts.prompt import PromptTemplate

prompt = PromptTemplate(
    template="Generate me a list of {count} catchy and unique domain names for a {business}",
    input_variables=["count", "business"]
)

result = openai_model.invoke(prompt.invoke({"count": 7, "business": "Shawarma Joint"}))
print(result)
```
# Prompts with Structured Output
Langchain allows for structured output using Pydantic.
```python
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator

# Define your desired data structure
class DomainNameList(BaseModel):
    domain_names: list[str] = Field(description="A list of domain names.")

parser = PydanticOutputParser(pydantic_object=DomainNameList)

prompt = PromptTemplate(
    template="Generate me a list of {count} domain names for a {business}. Output format: {format_instructions}",
    input_variables=["count", "business"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | openai_model | parser
output = chain.invoke({"count": 6, "business": "Chipotle joint"})
print(output)
print(type(output))
```
## Embeddings
Langchain supports embeddings with various LLMs.

```python
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
mistral_embeddings = OllamaEmbeddings(model="mistral")
result = openai_embeddings.embed_query("what are the SOLID principles in programming?")
print(result)
```

## Indexes / Vector Databases
Langchain integrates with vector databases for efficient retrieval.

```python
import os
from langchain_community.document_loaders import ConfluenceLoader

loader = ConfluenceLoader(
    url="https://accountname.atlassian.net/wiki", username="username", api_key=os.environ["ATLASSIAN_API_KEY"],
    space_key="confluence_space",
    include_attachments=False,
    limit=5
)

from langchain_chroma import Chroma
db = Chroma(persist_directory="./chroma_db", embedding_function=mistral_embeddings, collection_name="confluence_docs")
db.similarity_search("What is Resources are available?")
```

## Retrieval Augmented Generation (RAG)
RAG is a technique for augmenting LLM knowledge with additional data.

```python
from langchain import hub

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
for message in retrieval_qa_chat_prompt.messages:
    print(message)
    
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

retriever = db.as_retriever()
combine_docs_chain = create_stuff_documents_chain(
    mistral_model, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
result = retrieval_chain.invoke({"input": "What tools and resources are available in digital partnerships space? Give a list of key tools and resources."})
for message in result:
    if isinstance(result[message], list):
        print(message + ": " + ', '.join(map(str, result[message])))
    else:
        print(message + ": " + str(result[message]))
```

## Agents and Tools
Agents use a language model to choose a sequence of actions.

```python
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)  # Initialize the chat LLM
instructions = """You are an assistant."""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

tool = TavilySearchResults()  # Search engine tool
tavily_tool = TavilySearchResults()
tools = [tavily_tool]

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)
agent_executor.invoke({"input": "What is the weather in Burlington Ontario?"})
```
This guide should provide you with a solid foundation to start exploring and developing with Langchain. For more detailed documentation, visit the official [Langchain documentation](https://python.langchain.com/v0.2/docs/introduction/).


