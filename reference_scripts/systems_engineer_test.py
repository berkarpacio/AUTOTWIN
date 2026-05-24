from DesignOS.mbse.systems_engineering_agent import run_systems_engineering_agent
from DesignOS.workflow.engine import WorkflowContext
from DesignOS.workflow.engine import load_workflow
from DesignOS.workflow.engine import ADAPTER_REGISTRY
from DesignOS.workflow.engine import _resolve_input
from DesignOS.workflow.engine import _assign_output
from typing import Dict, Any

# Adapter imports
from DesignOS.adapters import (
    aero_opt_fw_adapter,
    prelim_sizing_mdo_adapter
)


HERO_PROMPT = "Placeholder prompt"
workflow_yaml_path="DesignOS/workflow/workflows/hero_quadplane.yaml"

def main():

    project_id = "hero_quad"

    campaign = run_systems_engineering_agent(project_id=project_id, user_prompt=HERO_PROMPT)
    wf_def = load_workflow(workflow_yaml_path)

    ctx = WorkflowContext(campaign=campaign)

    stage_1 = wf_def["stages"][1]

    payload: Dict[str, Any] = {}
    for input_name, ref in stage_1.get("inputs").items():
        payload[input_name] = _resolve_input(ref, ctx)

    result: Dict[str, Any] = aero_opt_fw_adapter.run(payload)
    for out_name, ref in stage_1.get("outputs").items():
        if out_name not in result:
            raise KeyError(f"Adapter '{aero_opt_fw_adapter}' did not return key '{out_name}'")
        _assign_output(ref, result[out_name], ctx)

    return ctx.snapshot()

if __name__ == "__main__":

    out = main()
    print(out["vars"])
    print(out["artifacts"])




