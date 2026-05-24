"""Utilities for visualizing or tabulating design campaign results."""

from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd


def campaign_parameters_dataframe(campaign) -> pd.DataFrame:
    """
    Build a single table of final campaign parameters.

    The output includes global parameters and component parameters, with each
    row carrying a parameter tag, units, and current value.
    """
    rows: List[Dict[str, Any]] = []

    for parameter in (
        campaign.global_parameters.known_variables
        + campaign.global_parameters.optimization_variables
        + campaign.global_parameters.computed_variables
    ):
        rows.append(
            {
                "group": "global",
                "component_id": None,
                "parameter_tag": parameter.tag,
                "units": parameter.units,
                "value": parameter.value,
            }
        )

    for component in campaign.components:
        for parameter in (
            component.known_variables
            + component.optimization_variables
            + component.computed_variables
        ):
            rows.append(
                {
                    "group": "component",
                    "component_id": component.id,
                    "parameter_tag": parameter.tag,
                    "units": parameter.units,
                    "value": parameter.value,
                }
            )
    
    for scenario in campaign.scenarios:
        for parameter in (
            scenario.known_variables
            + scenario.computed_variables
        ):
            rows.append(
                {
                    "group": "scenario",
                    "parameter_tag": parameter.tag,
                    "units": parameter.units,
                    "value": parameter.value,
                }
            )


    dataframe = pd.DataFrame(
        rows,
        columns=["group", "component_id", "parameter_tag", "units", "value"],
    )
    return dataframe.sort_values(
        by=["group", "component_id", "parameter_tag"], na_position="first"
    ).reset_index(drop=True)
