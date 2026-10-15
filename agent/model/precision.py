"""Versioned experimental precision contracts for policy execution."""
from dataclasses import dataclass
import torch
from agent.types.observation import ObservationBatch

@dataclass(frozen=True)
class PrecisionContract:
    mode: str
    quantize_observation: bool
    encoder_amp: bool
    scorer_amp: bool
    gru_amp: bool
    # The effects ragged projection is the only known batch-composition
    # sensitive FP16 GEMM on the V100 Mode-D path.  Keeping this narrow
    # projection in FP32 preserves AMP for the much larger dense encoder while
    # making its result independent of the number of packed ragged rows.
    ragged_effects_fp32: bool = False
    @property
    def identity(self) -> str:
        return f"pa3-precision-{self.mode}-v2"
    def prepare_observation(self, observation: ObservationBatch) -> ObservationBatch:
        if not self.quantize_observation:
            return observation
        floats = observation.floats.half().float()
        if bool((~torch.isfinite(floats)&observation.float_known).any()):
            raise ValueError("known observation overflows V2 FP16 contract")
        return ObservationBatch(
            token_mask=observation.token_mask, categories=observation.categories,
            category_known=observation.category_known, floats=floats,
            float_known=observation.float_known, flags=observation.flags,
            flag_known=observation.flag_known, role_ids=observation.role_ids,
            side_ids=observation.side_ids, layout=observation.layout,
            schema_version=observation.schema_version,
            contract_hash=observation.contract_hash,
            ragged={name:section.quantized(name) for name,section in observation.ragged.items()},
        )._remember_token_limit(observation.validated_token_limit)

_CONTRACTS = {
    "A": PrecisionContract("A", False, False, False, False),
    "B": PrecisionContract("B", False, True, True, True),
    "C": PrecisionContract("C", True, True, True, True),
    "D": PrecisionContract("D", True, True, False, False, True),
}
def get_precision_contract(mode: str) -> PrecisionContract:
    try:
        return _CONTRACTS[str(mode).upper()]
    except KeyError as exc:
        raise ValueError(f"unknown precision mode {mode!r}; expected A, B, C, or D") from exc
