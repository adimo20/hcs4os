# Classifiers

The four classification entry points. Each `Base…` class takes its configuration as
constructor arguments; each thin subclass looks the same configuration up in a registry by
dataset name.

## Hierarchy traversal

::: rac4os.agents.HierarchicalNavigation.BaseAgent.BaseHierarchicalNavigationAgent

::: rac4os.agents.HierarchicalNavigation.HierarchicalNavigationAgent.HierarchicalNavigationAgent

## Agentic RAG

::: rac4os.agents.RAG.BaseAgentRAG.BaseRAGAgent

::: rac4os.agents.RAG.RAGAgent.RAGAgent

## Linear RAG

::: rac4os.rag.BaseRAG.BaseRAGCoT

::: rac4os.rag.RAG.RAGChainOfThought

## Rewrite-Retrieve-Read

::: rac4os.rag.RAG.QueryRephraser

::: rac4os.rag.RAG.RRR_RAGChainOfThought
