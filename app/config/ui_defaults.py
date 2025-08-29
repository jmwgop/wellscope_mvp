# app/config/ui_defaults.py

from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass(frozen=True)
class UIDefaults:
    # Vector building defaults
    vector_months: int = 24
    vector_stream: str = "oil"
    vector_normalize: str = "q_over_qmax"
    boe_gas_factor: float = 6.0
    
    # Clustering defaults - optimized for production shape analysis
    min_cluster_size: int = 8  # Reasonable for production curve clustering
    min_samples: Optional[int] = None
    clustering_metric: str = "cosine"  # BEST for production curve shape similarity
    dbscan_eps: float = 0.5
    use_hdbscan: bool = True  # Superior for production curve clustering
    
    # Data-aware clustering defaults
    default_group_size_preference: str = "medium"  # "small", "medium", "large"
    default_clustering_sensitivity: str = "balanced"  # "loose", "balanced", "strict"
    
    # Universal scaling parameters
    small_dataset_threshold: int = 50     # Switch to simpler algorithms
    medium_dataset_threshold: int = 500   # Optimal HDBSCAN range
    large_dataset_threshold: int = 2000   # May need performance considerations
    
    # Confidence thresholds
    min_confidence_warning: float = 0.4   # Show warning if confidence < 40%
    good_confidence_threshold: float = 0.7  # Show success if confidence >= 70%
    
    # UMAP projection defaults
    umap_n_components: int = 2
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"
    random_state: int = 42
    
    # Filter defaults
    default_formations: List[str] = None
    default_completion_year_range: Tuple[Optional[int], Optional[int]] = (2018, 2024)
    default_lateral_range: Tuple[Optional[float], Optional[float]] = (5000.0, 20000.0)
    default_min_months_produced: int = 6
    
    # UI parameters
    similarity_threshold: float = 0.7
    max_file_size_mb: int = 100
    default_page_size: int = 20
    
    def __post_init__(self):
        if self.default_formations is None:
            object.__setattr__(self, 'default_formations', ["EAGLEFORD", "AUSTIN CHALK", "BUDA"])

# Global instance
DEFAULTS = UIDefaults()