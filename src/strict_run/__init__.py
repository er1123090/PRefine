"""Synthetic-fixture-safe V6 strict-run primitives."""

from .canonical import StrictRunError, canonical_json, sha256_bytes, validate_sealed_run_id
from .checkpoint import (
    CheckpointResult,
    prepare_checkpoint_directory,
    publish_blocked,
    publish_checkpoint,
    validate_checkpoint_payload,
    validate_final_tree,
)
from .cp45_semantic import (
    CP3SemanticPublications,
    CP4SemanticPublications,
    validate_cp4_semantics,
    validate_cp5_semantics,
)
from .external import (
    ExternalDocument,
    ExternalOutputEvidence,
    ExternalOutputReservation,
    ExternalTranscriptRecord,
    ExternalTranscriptRegistry,
    load_pre_reservation_document,
)
from .publication import ArtifactEvidence, Publication, StagePublisher, transcript_hash
from .reservation import (
    AcceptanceResult,
    ReservationInputs,
    ReservationResult,
    accept_reservation,
    publish_owner_binding,
    reserve_strict_run,
)
from .semantics import validate_cp2_artifact_structure
from .writer_policy import WriterIdentity, WriterPolicy, default_writer_policy

__all__ = [
    "AcceptanceResult",
    "ArtifactEvidence",
    "CheckpointResult",
    "CP3SemanticPublications",
    "CP4SemanticPublications",
    "ExternalDocument",
    "ExternalOutputEvidence",
    "ExternalOutputReservation",
    "ExternalTranscriptRecord",
    "ExternalTranscriptRegistry",
    "Publication",
    "ReservationInputs",
    "ReservationResult",
    "StagePublisher",
    "StrictRunError",
    "WriterIdentity",
    "WriterPolicy",
    "accept_reservation",
    "canonical_json",
    "default_writer_policy",
    "load_pre_reservation_document",
    "prepare_checkpoint_directory",
    "publish_blocked",
    "publish_checkpoint",
    "publish_owner_binding",
    "reserve_strict_run",
    "sha256_bytes",
    "transcript_hash",
    "validate_checkpoint_payload",
    "validate_cp2_artifact_structure",
    "validate_cp4_semantics",
    "validate_cp5_semantics",
    "validate_final_tree",
    "validate_sealed_run_id",
]
