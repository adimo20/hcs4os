# Retrieval prompts

The ReAct variants (`RAGAgent`) drive a search-and-refine loop. The
ChainOfThought variants (`RAGChainOfThought`, `RRR_RAGChainOfThought`) reason once
over a fixed candidate list. Both insist the final code appear verbatim among the
retrieved records.

## Search agent (ReAct)

::: hcs4os.agents.RAG.registry.CoicopRAGSearchAgentSignature
::: hcs4os.agents.RAG.registry.IcatusRAGSearchAgentSignature
::: hcs4os.agents.RAG.registry.SeaRAGSearchAgentSignature

## Chain of thought

::: hcs4os.rag.registry.CoicopRAGChainOfThoughtSignature
::: hcs4os.rag.registry.IcatusRAGChainOfThoughtSignature
::: hcs4os.rag.registry.SeaRAGChainOfThoughtSignature
