"""Shortcut-suppression modules shared across model owners.

Includes T18's Lung-Region Attention Module (Candidate A), built on
DenseNet121 and designed to be backbone-agnostic for reuse in T23
(applying the winning module to ResNet50), and T27's calibration harness
(calibration.py) -- architecture-agnostic, used by every model owner for
T28, not T18-specific. Import from here, not from `src.modules.<submodule>`
directly, matching this repo's convention for `src.utils` and
`src.datasets` (see CLAUDE.md).
"""

from .attention_metrics import (
    attention_dice,
    attention_entropy,
    attention_iou,
    background_attention,
    energy_inside_lung,
    ilar,
)
from .calibration import (
    TemperatureScaler,
    brier_score,
    calibration_report,
    collect_logits,
    expected_calibration_error,
    fit_temperature,
    plot_reliability_diagram,
    reliability_diagram_data,
)
from .comparison import (
    build_comparison_table,
    build_per_image_predictions,
    check_acceptance_criteria,
    stratified_cam_subset,
    summarize_multiseed,
)
from .counterfactual import counterfactual_stability, evaluate_counterfactual_robustness, perturb_background
from .efficiency_check import check_module_efficiency
from .figures import plot_attention_grid, plot_heatmap_row, select_heatmap_images
from .gradcam import cam_for, get_taps
from .lung_attention import (
    CBAMSpatialAttention,
    DenseNetLungAttention,
    LogitsOnly,
    LungRegionAttention,
    attention_guidance_loss,
    background_suppression_loss,
    build_model,
    compute_total_loss,
    freeze_backbone,
    print_trainable_parameters,
    unfreeze_final_blocks,
)
from .training import best_history_row, build_optimizer, build_scheduler, evaluate, run_epoch, run_full_arm, train_phase
from .statistics import build_statistics_report, mcnemar_exact_from_correctness, paired_wilcoxon
from .trustworthiness import (
    BACKBONE_CONFIGS,
    CNN_BACKBONES,
    build_master_row,
    default_config_path,
    load_guided_config,
    make_pair_configs,
    resolve_guided_hparams,
    seed_output_dir,
    select_guided_candidate,
    summarize_multiseed_master,
    upsert_master_comparison,
    validate_backbone_config,
    validation_selection_row,
    write_master_comparison,
    write_validation_selection_table,
)

__all__ = [
    "LungRegionAttention",
    "CBAMSpatialAttention",
    "DenseNetLungAttention",
    "LogitsOnly",
    "build_model",
    "attention_guidance_loss",
    "background_suppression_loss",
    "compute_total_loss",
    "freeze_backbone",
    "unfreeze_final_blocks",
    "print_trainable_parameters",
    "ilar",
    "attention_iou",
    "attention_dice",
    "attention_entropy",
    "background_attention",
    "energy_inside_lung",
    "cam_for",
    "get_taps",
    "run_epoch",
    "train_phase",
    "evaluate",
    "build_optimizer",
    "build_scheduler",
    "best_history_row",
    "run_full_arm",
    "stratified_cam_subset",
    "build_per_image_predictions",
    "build_comparison_table",
    "check_acceptance_criteria",
    "select_heatmap_images",
    "plot_heatmap_row",
    "plot_attention_grid",
    "check_module_efficiency",
    "summarize_multiseed",
    "perturb_background",
    "counterfactual_stability",
    "evaluate_counterfactual_robustness",
    "TemperatureScaler",
    "collect_logits",
    "expected_calibration_error",
    "brier_score",
    "reliability_diagram_data",
    "fit_temperature",
    "plot_reliability_diagram",
    "calibration_report",
    "CNN_BACKBONES",
    "BACKBONE_CONFIGS",
    "default_config_path",
    "validate_backbone_config",
    "seed_output_dir",
    "load_guided_config",
    "resolve_guided_hparams",
    "make_pair_configs",
    "validation_selection_row",
    "write_validation_selection_table",
    "build_master_row",
    "write_master_comparison",
    "upsert_master_comparison",
    "summarize_multiseed_master",
    "select_guided_candidate",
    "mcnemar_exact_from_correctness",
    "paired_wilcoxon",
    "build_statistics_report",
]
