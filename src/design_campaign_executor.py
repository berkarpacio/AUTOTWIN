"""This script includes methods for executing a design campaign"""

# --- 0) Imports ---
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import importlib


@dataclass
class StageResult:
    stage_id: str
    outputs: Dict[str, Any]


class ParameterStore:
    """
    Unified parameter store:
      - supports lookup by tag: "S_wing"
      - supports optional namespaces: "MAIN_WING.S_wing", "MOO-MISSION.g"
      - supports bounds retrieval: "S_wing_bounds" or bounds("S_wing")
    """

    def __init__(self):
        self.values: Dict[str, Any] = {}
        self.bounds: Dict[str, List[float]] = {}

    @staticmethod
    def _normalize_key(key: Any) -> str:
        return str(key).strip()

    def set_value(self, key: str, value: Any) -> None:
        self.values[self._normalize_key(key)] = value

    def get_value(self, key: str) -> Any:
        k = self._normalize_key(key)
        if k in self.values and self.values[k] is not None:
            return self.values[k]
        # fallback: if "COMP.tag" not found, try "tag"
        if "." in k:
            short = k.split(".", 1)[1]
            if short in self.values and self.values[short] is not None:
                return self.values[short]
        # As a last resort, return the raw key even if it's None to preserve old behavior
        if k in self.values:
            return self.values[k]
        raise KeyError(f"Parameter not found: {key}")

    def set_bounds(self, tag: str, bounds: List[float]) -> None:
        self.bounds[self._normalize_key(tag)] = bounds

    def get_bounds(self, tag: str) -> List[float]:
        t = self._normalize_key(tag)
        if t in self.bounds:
            return self.bounds[t]
        # fallback for "COMP.tag"
        if "." in t:
            short = t.split(".", 1)[1]
            if short in self.bounds:
                return self.bounds[short]
        raise KeyError(f"Bounds not found for: {tag}")

    def resolve_input_token(self, token: str) -> Tuple[str, Any]:
        """
        Converts an input token from YAML into a (name, value) pair
        passed into a model function.
        Supported:
          - "S_wing" -> ("S_wing", value)
          - "MAIN_WING.S_wing" -> ("S_wing", value)   (name becomes last segment)
          - "S_wing_bounds" -> ("S_wing_bounds", bounds)
          - "bounds(S_wing)" -> ("S_wing_bounds", bounds)
        """
        token = token.strip()

        # bounds(...) syntax
        if token.startswith("bounds(") and token.endswith(")"):
            inner = token[len("bounds("):-1].strip()
            return f"{inner}_bounds", self.get_bounds(inner)

        # explicit *_bounds
        if token.endswith("_bounds"):
            base = token[:-len("_bounds")]
            name = token.split(".", 1)[-1]
            return name, self.get_bounds(base)

        # normal param
        name = token.split(".", 1)[-1]  # last segment
        return name, self.get_value(token)

    def update_from_output(self, output_dict: Dict[str, Any]) -> None:
        """
        Writes outputs back into the store, by matching keys.
        """
        for k, v in output_dict.items():
            # always store by raw key
            self.set_value(k, v)

    def update_from_declared_outputs(self, declared_outputs: Iterable[str], output_dict: Dict[str, Any]) -> None:
        """
        Bind model-returned values to the output tokens declared by the YAML.

        Models often return short keys such as "S_wing", while workflow outputs
        are component-scoped as "MAIN_WING.S_wing". Store both forms so later
        stages can depend on either spelling.
        """
        for token in declared_outputs:
            token = self._normalize_key(token)
            short = token.split(".", 1)[-1]

            if token in output_dict:
                value = output_dict[token]
            elif short in output_dict:
                value = output_dict[short]
            else:
                continue

            self.set_value(token, value)
            self.set_value(short, value)


def load_callable(entrypoint: str):
    """
    entrypoint: "libraries.aerospace.optimization.small_UAV_eVTOL_MOO:small_UAV_eVTOL_MOO"
    or          "optimization:small_UAV_eVTOL_MOO" (if your module path is set)
    """
    if ":" not in entrypoint:
        raise ValueError(f"Invalid entrypoint '{entrypoint}'. Use 'module.submodule:function_name'.")

    module_name, func_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if func is None:
        raise ImportError(f"Function '{func_name}' not found in module '{module_name}'.")
    return func


class WorkflowExecutor:
    def __init__(self, campaign):
        self.campaign = campaign
        self.store = ParameterStore()

    def build_store(self) -> None:
        """
        Populate store with:
          - known values
          - bounds for optimization variables
          - existing computed variables (as None)
        """
        # globals
        for p in self.campaign.global_parameters.known_variables:
            self.store.set_value(p.tag, p.value)
        for p in self.campaign.global_parameters.optimization_variables:
            if p.bounds is not None:
                self.store.set_bounds(p.tag, p.bounds)
            if p.value is not None:
                self.store.set_value(p.tag, p.value)
        for p in self.campaign.global_parameters.computed_variables:
            if p.value is not None:
                self.store.set_value(p.tag, p.value)

        # components (store namespaced and un-namespaced, for flexibility)
        for c in self.campaign.components:
            for p in (c.known_variables + c.optimization_variables + c.computed_variables):
                if p.bounds is not None:
                    self.store.set_bounds(p.tag, p.bounds)
                if p.value is not None:
                    self.store.set_value(p.tag, p.value)
                # Namespaced key, helps avoid collisions later
                self.store.set_value(f"{c.id}.{p.tag}", p.value)

        # scenarios
        for s in self.campaign.scenarios:
            for p in s.known_variables:
                self.store.set_value(p.tag, p.value)
                self.store.set_value(f"{s.id}.{p.tag}", p.value)
            for p in s.computed_variables:
                if p.value is not None:
                    self.store.set_value(p.tag, p.value)
                    self.store.set_value(f"{s.id}.{p.tag}", p.value)

    def _stage_inputs(self, stage) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}
        missing: List[str] = []
        for token in stage.inputs:
            try:
                k, v = self.store.resolve_input_token(token)
                inputs[k] = v
            except KeyError:
                missing.append(token)

        if missing:
            raise KeyError(
                f"Stage '{stage.id}' is missing inputs: {missing}\n"
                f"Tip: check YAML input tokens and tags/bounds naming."
            )
        return inputs

    def _write_back_to_campaign(self) -> None:
        """
        Sync store values back into campaign Parameter objects (so UI can render campaign state).
        """
        # globals
        for p in (self.campaign.global_parameters.known_variables
                  + self.campaign.global_parameters.optimization_variables
                  + self.campaign.global_parameters.computed_variables):
            if p.tag in self.store.values:
                p.value = self.store.values[p.tag]

        # components
        for c in self.campaign.components:
            for p in (c.known_variables + c.optimization_variables + c.computed_variables):
                namespaced_tag = f"{c.id}.{p.tag}"
                if namespaced_tag in self.store.values and self.store.values[namespaced_tag] is not None:
                    p.value = self.store.values[namespaced_tag]
                elif p.tag in self.store.values:
                    p.value = self.store.values[p.tag]
                # keep namespaced keys in sync for stage inputs like MAIN_WING.S_wing
                self.store.set_value(f"{c.id}.{p.tag}", p.value)

        # scenarios
        for s in self.campaign.scenarios:
            for p in (s.known_variables + s.computed_variables):
                if p.tag in self.store.values:
                    p.value = self.store.values[p.tag]
                self.store.set_value(f"{s.id}.{p.tag}", p.value)

    def run(self) -> List[StageResult]:
        self.build_store()

        results: List[StageResult] = []

        for stage in self.campaign.workflow.stages:
            func = load_callable(stage.model)
            stage_input_dict = self._stage_inputs(stage)

            out = func(stage_input_dict)

            # normalize output to dict
            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
                # e.g. (aircraft_object, geometric_parameters)
                output_dict = dict(out[1])
                # optionally store object too
                output_dict[f"{stage.id}__object"] = out[0]
            elif isinstance(out, dict):
                output_dict = out
            else:
                raise TypeError(
                    f"Stage '{stage.id}' returned unsupported type: {type(out)}. "
                    f"Return dict or (obj, dict)."
                )

            self.store.update_from_output(output_dict)
            self.store.update_from_declared_outputs(stage.outputs, output_dict)
            missing_outputs = [
                token
                for token in stage.outputs
                if token not in self.store.values
                and token.split(".", 1)[-1] not in self.store.values
            ]
            if output_dict.get("status") not in (None, "ok") and missing_outputs:
                raise RuntimeError(
                    f"Stage '{stage.id}' returned status '{output_dict.get('status')}' "
                    f"without required outputs: {missing_outputs}. "
                    f"{output_dict.get('message', '')}"
                )
            self._write_back_to_campaign()

            results.append(StageResult(stage_id=stage.id, outputs=output_dict))

        return results
