# Prompt Overview

The traversal agent reads five pieces of prose: the **signature docstring**, which is the
system prompt, and **four tool descriptions**, one per navigation operation. All of them
are rendered below from the installed package, so they are exactly what the model
receives.

## Signature table

| Dataset               | Signature                                   |
| --------------------- | ------------------------------------------- |
| `COICOP_2018`         | `CoicopHierarchicalSearchAgentSignature`    |
| `ICATUS_2016`         | `IcatusHierarchicalSearchAgentSignature`    |
| `NAF_rev2`            | `NafHierarchicalSearchAgentSignature`       |
| `SEA_2021`            | `SeaHierarchicalSearchAgentSignature`       |
| `SEA_2021_REPHRASED`  | `SeaHierarchicalSearchAgentSignature`       |

Each entry in `HierarchicalNavigationAgent.CONFIGS` bundles the signature with the four
tool descriptions:

```python
"SEA_2021": {
    "signature": SeaHierarchicalSearchAgentSignature,
    "get_root_categories_doc": "...",
    "get_children_doc": "...",
    "get_code_doc": "...",
    "get_parent_doc": "...",
}
```

The keys are the base constructor's parameter names, so renaming a parameter means
renaming the key.

## Commonly used instructions

| Element              | What it does                                                      |
| -------------------- | ----------------------------------------------------------------- |
| Role and task        | names the taxonomy and the unit being classified                  |
| Descent protocol     | start at the roots, compare, descend one level at a time          |
| Comparison rule      | read *every* child before choosing a branch                       |
| Verification step    | check what a candidate covers before committing                   |
| Backtracking rule    | go back up when nothing at a level fits                           |
| Exclusion handling   | follow pointers to other categories instead of forcing a fit      |
| Target level         | which level the descent should end at                             |
| Grounding constraint | never emit a code that did not come from a tool call              |

The comparison rule and the backtracking rule are the ones that earn their space. They
address the approach's [failure mode](how-it-works.md#the-failure-mode) — an early wrong
turn that nothing downstream can recover — and an agent not given them will force the
item into the best of a wrong branch.

## The tool descriptions

Each should state what the tool returns *and when to use it*:

| Parameter                 | Should say                                              |
| ------------------------- | ------------------------------------------------------- |
| `get_root_categories_doc` | the entry point; call it first                           |
| `get_children_doc`        | descend exactly one level                                |
| `get_code_doc`            | verify a candidate before answering                      |
| `get_parent_doc`          | back out of a branch that turned out to be wrong         |

```python
get_children_doc = (
    "Return the child categories of a code, with their descriptions and what they "
    "include. Use it to descend exactly one level at a time."
)
```

!!! tip

    Use your own level names. If your taxonomy calls them divisions, groups and classes,
    say so here and in the signature; the model is navigating your structure, and naming
    it as your documentation does makes the trajectory easier to read and debug.

## Writing your own

See [Adapting Overview](adapting.md#signature) for a complete signature plus the four tool
descriptions written for a custom taxonomy. Changing a prompt requires no registry edit —
pass the signature and the four strings straight to
`BaseHierarchicalNavigationAgent` and A/B two versions in one script.

## Bundled signatures

::: rac4os.agents.HierarchicalNavigation.registry.CoicopHierarchicalSearchAgentSignature

::: rac4os.agents.HierarchicalNavigation.registry.IcatusHierarchicalSearchAgentSignature

::: rac4os.agents.HierarchicalNavigation.registry.NafHierarchicalSearchAgentSignature

::: rac4os.agents.HierarchicalNavigation.registry.SeaHierarchicalSearchAgentSignature
