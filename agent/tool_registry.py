"""
agent/tool_registry.py
========================
A lightweight, dependency-free registry that lets each specialist model
module (`models/*.py`) advertise itself declaratively. The orchestrator
selects tools purely from this registry, so adding a new specialist is a
matter of decorating its entry-point function with `@register_tool(...)`
— no changes to `orchestrator.py` are required.

Design goals
------------
* **Deterministic selection**: registry lookup is by exact
  (task_type, input_config) match, never by a black-box model call, so
  the orchestrator's routing decision is itself auditable.
* **Interface uniformity**: every registered tool must accept
  `(images: List[RasterImage], query: str, **kwargs) -> ToolResult` so
  the orchestrator can invoke any tool identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from core.geo_io import RasterImage


@dataclass
class ToolResult:
    """Uniform output contract for every specialist tool."""
    task_type: str
    text_answer: str
    confidence: float                       # 0.0 - 1.0
    bounding_boxes: List[dict] = field(default_factory=list)   # [{"xmin","ymin","xmax","ymax","label","score"}]
    masks: List[dict] = field(default_factory=list)            # [{"label","array": np.ndarray(bool), "score"}]
    evidence: Dict[str, Any] = field(default_factory=dict)     # numeric evidence (indices, stats)
    model_used: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class ToolSpec:
    name: str
    task_types: List[str]
    input_configs: List[str]
    fn: Callable[..., ToolResult]
    description: str = ""


_REGISTRY: Dict[str, ToolSpec] = {}


def register_tool(name: str, task_types: List[str], input_configs: List[str],
                   description: str = "") -> Callable:
    """
    Decorator used by each `models/*.py` module to register its
    entry-point callable into the global tool registry.

    Example
    -------
    @register_tool("rs_vqa_caption", task_types=[TASK_VQA, TASK_CAPTION],
                    input_configs=[INPUT_CONFIG_SINGLE])
    def run(images, query, **kwargs) -> ToolResult:
        ...
    """
    def decorator(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        _REGISTRY[name] = ToolSpec(
            name=name, task_types=task_types, input_configs=input_configs,
            fn=fn, description=description or fn.__doc__ or "",
        )
        return fn
    return decorator


def get_tool(name: str) -> Optional[ToolSpec]:
    return _REGISTRY.get(name)


def find_tools(task_type: str, input_config: str) -> List[ToolSpec]:
    """Return all registered tools that can service `task_type` under
    `input_config`, in registration order (deterministic)."""
    return [
        spec for spec in _REGISTRY.values()
        if task_type in spec.task_types and input_config in spec.input_configs
    ]


def all_tools() -> Dict[str, ToolSpec]:
    return dict(_REGISTRY)


def registry_summary() -> List[dict]:
    """JSON-serialisable summary of the registry, for the GUI's pipeline
    graph and the audit trace."""
    return [
        {"name": s.name, "task_types": s.task_types,
         "input_configs": s.input_configs, "description": s.description}
        for s in _REGISTRY.values()
    ]
