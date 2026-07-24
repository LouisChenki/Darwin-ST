"""创造层 (Creation Layer) —— Tier-2 组合式创造 (Level-2)。

把 P3-a 跨域知识接上 LLM 融合, 合成新算子。组合式创造 = 把多个跨域机制化学融合成
新算子, 由零初始化残差守卫 + 验证 harness + 真实评测守护。
详见 docs/TIER2_RESEARCH_FINDINGS.md §7。
"""

from darwin_st.creation.fusion import ZeroInitResidualFusion, GatedFusion
from darwin_st.creation.validation import ValidationConfig, ValidationReport, validate_operator
from darwin_st.creation.contracts import (
    COMPOSITION_OPS,
    FusionRequest,
    FusionPlan,
    SynthesizedOperator,
)
from darwin_st.creation.llm import LLMClient, MockLLM, OpenAICompatLLM
from darwin_st.creation.aider_backend import AiderConfig, AiderBackend
from darwin_st.creation.registry import OperatorRegistry, SYNTH_PREFIX, family_of, version_of
from darwin_st.creation.refiner import (
    RefineConfig,
    OperatorRefiner,
    REFINE_MODES,
    REFINE_MODE_WEIGHTS,
)
from darwin_st.creation.creation_loop import (
    CreationConfig,
    CreationOutcome,
    CreationLoop,
    diagnose_bottleneck,
)
from darwin_st.creation.creation_archive import CreationArchive, CreationRecord
from darwin_st.creation.synthesizer import (
    SynthesisConfig,
    SynthesisResult,
    OperatorSynthesizer,
    extract_code,
    exec_operator_code,
)

__all__ = [
    "ZeroInitResidualFusion",
    "GatedFusion",
    "ValidationConfig",
    "ValidationReport",
    "validate_operator",
    "COMPOSITION_OPS",
    "FusionRequest",
    "FusionPlan",
    "SynthesizedOperator",
    "LLMClient",
    "MockLLM",
    "OpenAICompatLLM",
    "AiderConfig",
    "AiderBackend",
    "OperatorRegistry",
    "SYNTH_PREFIX",
    "family_of",
    "version_of",
    "RefineConfig",
    "OperatorRefiner",
    "REFINE_MODES",
    "REFINE_MODE_WEIGHTS",
    "CreationConfig",
    "CreationOutcome",
    "CreationLoop",
    "diagnose_bottleneck",
    "CreationArchive",
    "CreationRecord",
    "SynthesisConfig",
    "SynthesisResult",
    "OperatorSynthesizer",
    "extract_code",
    "exec_operator_code",
]
