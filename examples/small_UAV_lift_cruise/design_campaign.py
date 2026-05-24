from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Literal, Tuple, Union

import yaml as pyyaml

@dataclass
class Verification:
    method: Literal["analysis", "test", "inspection", "demonstration"]
    model: str # e.g. small_UAV_eVTOL_MOO.py
    check: str # e.g. "10 m/s <= V_cruise <= 30 m/s"

@dataclass
class Requirement:
    id: str
    statement: str
    severity: Optional[str]
    verify: Verification

@dataclass
class Parameter:
    tag: str
    type: str
    units: Optional[str]
    value: Optional[float]
    bounds: Optional[List[float]]

@dataclass
class GlobalParameters:
    optimization_variables: List[Parameter]
    known_variables: List[Parameter]
    computed_variables: List[Parameter]

@dataclass
class Component:
    id: str
    type: str
    optimization_variables: List[Parameter]
    known_variables: List[Parameter]
    computed_variables: List[Parameter]

@dataclass
class WorkflowStage:
    id: str
    description: Optional[str]
    model: str
    inputs: List[str]
    outputs: List[str]

@dataclass
class DesignWorkflow:
    id: str
    name: str
    stages: List[WorkflowStage]

@dataclass
class Scenario:
    id: str
    type: str
    known_variables: List[Parameter]
    computed_variables: List[Parameter]

    @property
    def known(self) -> List[Parameter]:
        return self.known_variables


@dataclass
class DesignCampaign:
    requirements: List[Requirement]
    global_parameters: GlobalParameters
    components: List[Component]
    workflow: DesignWorkflow
    scenarios: List[Scenario]

def _coerce_parameter_mapping(raw: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(raw, Mapping):
        return {str(k): dict(v or {}) for k, v in raw.items()}

    if isinstance(raw, list):
        coerced: Dict[str, Dict[str, Any]] = {}
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            tag = item.get("id") or item.get("tag")
            if tag is None:
                continue
            payload = dict(item)
            payload.pop("id", None)
            payload.pop("tag", None)
            coerced[str(tag)] = payload
        return coerced

    return {}


def _normalize_kind(kind: Optional[str]) -> str:
    text = str(kind or "").strip().lower()
    if "optim" in text:
        return "optimization"
    if "comput" in text:
        return "computed"
    if "known" in text or "knwon" in text or "const" in text or "fixed" in text:
        return "known"
    return "unknown"


def _build_parameter(tag: str, payload: Mapping[str, Any]) -> Parameter:
    return Parameter(
        tag=str(tag),
        type=str(payload.get("type", "float")),
        units=payload.get("units"),
        value=payload.get("value"),
        bounds=payload.get("bounds"),
    )


def _split_parameters(raw_parameters: Any) -> Tuple[List[Parameter], List[Parameter], List[Parameter]]:
    optimization_variables: List[Parameter] = []
    known_variables: List[Parameter] = []
    computed_variables: List[Parameter] = []

    parameter_map = _coerce_parameter_mapping(raw_parameters)
    for tag, payload in parameter_map.items():
        parameter = _build_parameter(tag, payload)
        bucket = _normalize_kind(payload.get("kind"))

        if bucket == "optimization":
            optimization_variables.append(parameter)
        elif bucket == "known":
            known_variables.append(parameter)
        elif bucket == "computed":
            computed_variables.append(parameter)
        else:
            if payload.get("bounds") is not None:
                optimization_variables.append(parameter)
            elif "value" in payload:
                known_variables.append(parameter)
            else:
                computed_variables.append(parameter)

    return optimization_variables, known_variables, computed_variables


def _read_yaml_source(yaml_input: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(yaml_input, dict):
        return yaml_input

    if isinstance(yaml_input, Path):
        text = yaml_input.read_text(encoding="utf-8")
    elif isinstance(yaml_input, str):
        path = Path(yaml_input)
        text = path.read_text(encoding="utf-8") if path.exists() else yaml_input
    else:
        raise TypeError("yaml must be a dict, path, or YAML string.")

    try:
        loaded = pyyaml.safe_load(text)
    except pyyaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ValueError("YAML content must be a mapping/object at the top level.")

    return loaded


def generate_design_campaign(yaml: Union[str, Path, Dict[str, Any]]) -> DesignCampaign:
    data = _read_yaml_source(yaml)

    requirements: List[Requirement] = []
    for req in data.get("requirements", []) or []:
        if not isinstance(req, Mapping):
            continue
        verify = req.get("verify", {}) or {}
        requirements.append(
            Requirement(
                id=str(req.get("id", "")),
                statement=str(req.get("statement", "")),
                severity=req.get("severity"),
                verify=Verification(
                    method=str(verify.get("method", "analysis")),
                    model=str(verify.get("model", "")),
                    check=str(verify.get("check", "")),
                ),
            )
        )

    global_ov, global_kv, global_cv = _split_parameters(data.get("parameters", {}))
    global_parameters = GlobalParameters(
        optimization_variables=global_ov,
        known_variables=global_kv,
        computed_variables=global_cv,
    )

    components: List[Component] = []
    for component in data.get("components", []) or []:
        if not isinstance(component, Mapping):
            continue
        ov, kv, cv = _split_parameters(component.get("parameters", {}))
        components.append(
            Component(
                id=str(component.get("id", "")),
                type=str(component.get("type", "")),
                optimization_variables=ov,
                known_variables=kv,
                computed_variables=cv,
            )
        )

    stages: List[WorkflowStage] = []
    for model in data.get("models", []) or []:
        if not isinstance(model, Mapping):
            continue

        inputs = [str(item) for item in model.get("inputs", []) or []]
        outputs: List[str] = []
        for output in model.get("outputs", []) or []:
            if isinstance(output, Mapping):
                outputs.append(str(output.get("name", "")))
            else:
                outputs.append(str(output))

        stages.append(
            WorkflowStage(
                id=str(model.get("id", "")),
                description=model.get("type"),
                model=str(model.get("entrypoint") or model.get("id", "")),
                inputs=inputs,
                outputs=outputs,
            )
        )

    system = data.get("system", {}) if isinstance(data.get("system"), Mapping) else {}
    workflow = DesignWorkflow(
        id=str(system.get("id", "design_workflow")),
        name=str(system.get("name", "Design Workflow")),
        stages=stages,
    )

    scenarios: List[Scenario] = []
    for scenario in data.get("scenarios", []) or []:
        if not isinstance(scenario, Mapping):
            continue
        scenario_ov, scenario_kv, scenario_cv = _split_parameters(scenario.get("parameters", {}))
        # Scenario schema does not expose optimization variables, keep them with known variables.
        scenarios.append(
            Scenario(
                id=str(scenario.get("id", "")),
                type=str(scenario.get("type", "")),
                known_variables=scenario_kv + scenario_ov,
                computed_variables=scenario_cv,
            )
        )

    return DesignCampaign(
        requirements=requirements,
        global_parameters=global_parameters,
        components=components,
        workflow=workflow,
        scenarios=scenarios,
    )
