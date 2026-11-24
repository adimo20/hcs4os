# Rephrasing prompts

Used by `QueryRephraser` in the Rewrite-Retrieve-Read pipeline. These restate a raw
item in the wording of the classification's own labels so that retrieval lands in
the right neighbourhood. They explicitly do **not** classify: no code may appear in
the output.

::: hcs4os.rag.registry.CoicopRephrasingSignature
::: hcs4os.rag.registry.IcatusRephrasingSignature
::: hcs4os.rag.registry.SeaRephrasingSignature
