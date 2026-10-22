"""Per-cohort extraction → common format (§8.3). The only package that knows the original tables."""

from perisim.data.common import INPUTS_SCHEMA, OUTCOMES_SCHEMA, STAYS_SCHEMA  # noqa: F401


def get_extractor(cohort: str):
    if cohort == "inspire":
        from perisim.data.inspire import InspireExtractor

        return InspireExtractor
    if cohort == "mimic":
        from perisim.data.mimic import MimicExtractor

        return MimicExtractor
    if cohort == "eicu":
        from perisim.data.eicu import EicuExtractor

        return EicuExtractor
    raise ValueError(f"Unknown cohort: {cohort}")
