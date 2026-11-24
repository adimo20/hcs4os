# Navigation prompts

Used by `HierarchicalNavigationAgent`. Each lays out a descent protocol: start at
the roots, compare every child, follow the `excludes` pointers instead of forcing a
fit, verify the leaf, and never emit a code that did not come from a tool.

::: hcs4os.agents.HierarchicalNavigation.registry.CoicopHierarchicalSearchAgentSignature
::: hcs4os.agents.HierarchicalNavigation.registry.IcatusHierarchicalSearchAgentSignature
::: hcs4os.agents.HierarchicalNavigation.registry.SeaHierarchicalSearchAgentSignature
::: hcs4os.agents.HierarchicalNavigation.registry.NafHierarchicalSearchAgentSignature
