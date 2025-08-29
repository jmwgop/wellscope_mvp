#!/usr/bin/env python3
"""
Advanced Clustering Experimentation Script
==========================================

This script systematically tests different clustering approaches on your session data
to understand why you're getting only 1 cluster and find better alternatives.

Usage:
    python scripts/experiment_clustering.py [session_id]

If no session_id provided, uses the latest session.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the existing session management
from app.services.session_persistence import get_session_manager

@dataclass
class ExperimentResult:
    """Container for clustering experiment results."""
    method: str
    params: Dict[str, Any]
    n_clusters: int
    n_noise: int
    silhouette_score: Optional[float]
    calinski_harabasz_score: Optional[float]
    davies_bouldin_score: Optional[float]
    labels: np.ndarray
    success: bool
    notes: str = ""

class ClusteringExperiment:
    """Main experimenter class for testing different clustering approaches."""
    
    def __init__(self, session_data: Dict[str, Any]):
        self.session_data = session_data
        self.vectors_df = session_data['vectors_df']
        self.api_col = [c for c in self.vectors_df.columns if not c.startswith('v')][0]
        self.vector_cols = [c for c in self.vectors_df.columns if c.startswith('v')]
        self.X = self.vectors_df[self.vector_cols].values
        self.n_wells = len(self.X)
        
        print(f"🔬 Initialized experiment with {self.n_wells} wells and {len(self.vector_cols)} features")
        print(f"📊 API column: {self.api_col}")
        print(f"📈 Vector columns: {self.vector_cols[:3]}...{self.vector_cols[-1]} ({len(self.vector_cols)} total)")
        
        # Calculate baseline similarity statistics
        self._calculate_similarity_stats()
        
    def _calculate_similarity_stats(self):
        """Calculate similarity statistics for the dataset."""
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        
        print("\n📈 Dataset Similarity Analysis:")
        
        # Cosine similarity
        cos_sim = cosine_similarity(self.X)
        upper_tri = np.triu_indices_from(cos_sim, k=1)
        cos_similarities = cos_sim[upper_tri]
        
        print(f"   Cosine Similarity: Mean={cos_similarities.mean():.3f}, "
              f"Min={cos_similarities.min():.3f}, Max={cos_similarities.max():.3f}")
        
        # Euclidean distances
        euc_dist = euclidean_distances(self.X)
        euc_distances = euc_dist[upper_tri]
        
        print(f"   Euclidean Distance: Mean={euc_distances.mean():.3f}, "
              f"Min={euc_distances.min():.3f}, Max={euc_distances.max():.3f}")
        
        # Distribution analysis
        self.similarity_stats = {
            'cosine_mean': float(cos_similarities.mean()),
            'cosine_std': float(cos_similarities.std()),
            'cosine_min': float(cos_similarities.min()),
            'euclidean_mean': float(euc_distances.mean()),
            'euclidean_std': float(euc_distances.std()),
            'euclidean_max': float(euc_distances.max()),
        }
        
        # High similarity threshold
        high_sim_threshold = 0.95
        high_sim_fraction = (cos_similarities > high_sim_threshold).mean()
        print(f"   Wells with >{high_sim_threshold*100}% similarity: {high_sim_fraction:.1%}")
        
    def _evaluate_clustering(self, labels: np.ndarray, method: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Evaluate clustering quality using multiple metrics."""
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            # Need at least 2 clusters for most metrics
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            if n_clusters < 2:
                return None, None, None
            
            # Filter out noise points for some metrics
            non_noise_mask = labels >= 0
            if non_noise_mask.sum() < 2:
                return None, None, None
                
            X_clean = self.X[non_noise_mask]
            labels_clean = labels[non_noise_mask]
            
            if len(set(labels_clean)) < 2:
                return None, None, None
            
            sil_score = silhouette_score(X_clean, labels_clean)
            ch_score = calinski_harabasz_score(X_clean, labels_clean)
            db_score = davies_bouldin_score(X_clean, labels_clean)
            
            return sil_score, ch_score, db_score
            
        except Exception as e:
            print(f"      Warning: Could not evaluate {method} clustering: {e}")
            return None, None, None
    
    def phase1_aggressive_current_methods(self) -> List[ExperimentResult]:
        """Phase 1: Test aggressive versions of current HDBSCAN/DBSCAN methods."""
        print("\n🚀 Phase 1: Aggressive Current Methods")
        results = []
        
        # Ultra-aggressive HDBSCAN tests
        hdbscan_configs = [
            {'min_cluster_size': 2, 'min_samples': 2, 'cluster_selection_method': 'leaf'},
            {'min_cluster_size': 3, 'min_samples': 2, 'cluster_selection_method': 'leaf'},
            {'min_cluster_size': 2, 'min_samples': 1, 'cluster_selection_method': 'eom'},
            {'min_cluster_size': 5, 'min_samples': 3, 'cluster_selection_epsilon': 0.01},
            {'min_cluster_size': 3, 'min_samples': 2, 'metric': 'cosine'},
            {'min_cluster_size': 4, 'min_samples': 2, 'metric': 'manhattan'},
        ]
        
        try:
            import hdbscan
            print("   Testing HDBSCAN variations...")
            
            for i, config in enumerate(hdbscan_configs):
                try:
                    clusterer = hdbscan.HDBSCAN(**config)
                    labels = clusterer.fit_predict(self.X)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = (labels == -1).sum()
                    
                    sil, ch, db = self._evaluate_clustering(labels, 'HDBSCAN')
                    
                    result = ExperimentResult(
                        method=f"HDBSCAN_{i+1}",
                        params=config,
                        n_clusters=n_clusters,
                        n_noise=n_noise,
                        silhouette_score=sil,
                        calinski_harabasz_score=ch,
                        davies_bouldin_score=db,
                        labels=labels,
                        success=n_clusters > 0
                    )
                    results.append(result)
                    
                    print(f"      HDBSCAN_{i+1}: {n_clusters} clusters, {n_noise} noise "
                          f"(sil={sil:.3f if sil else 'N/A'})")
                    
                except Exception as e:
                    print(f"      HDBSCAN_{i+1}: Failed - {e}")
                    
        except ImportError:
            print("   HDBSCAN not available, skipping...")
        
        # Aggressive DBSCAN tests with adaptive epsilon
        print("   Testing DBSCAN variations...")
        dbscan_configs = [
            {'eps': 0.1, 'min_samples': 2},
            {'eps': 0.2, 'min_samples': 3},
            {'eps': 0.3, 'min_samples': 2},
            {'eps': 0.05, 'min_samples': 2},
        ]
        
        # Add adaptive epsilon tests
        try:
            from sklearn.neighbors import NearestNeighbors
            from sklearn.cluster import DBSCAN
            
            # Calculate adaptive epsilon values
            for min_samples in [2, 3, 4]:
                nbrs = NearestNeighbors(n_neighbors=min_samples + 1).fit(self.X)
                distances, _ = nbrs.kneighbors(self.X)
                k_distances = np.sort(distances[:, min_samples])
                
                adaptive_eps = [
                    np.percentile(k_distances, 25),
                    np.percentile(k_distances, 50),
                    np.percentile(k_distances, 75),
                ]
                
                for eps in adaptive_eps:
                    if eps > 0:
                        dbscan_configs.append({
                            'eps': eps, 
                            'min_samples': min_samples,
                            'metric': 'euclidean'
                        })
                        
            # Test different metrics
            base_eps = np.percentile(k_distances, 50)
            for metric in ['cosine', 'manhattan']:
                dbscan_configs.append({
                    'eps': base_eps,
                    'min_samples': 3,
                    'metric': metric
                })
            
            for i, config in enumerate(dbscan_configs):
                try:
                    clusterer = DBSCAN(**config)
                    labels = clusterer.fit_predict(self.X)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = (labels == -1).sum()
                    
                    sil, ch, db = self._evaluate_clustering(labels, 'DBSCAN')
                    
                    result = ExperimentResult(
                        method=f"DBSCAN_{i+1}",
                        params=config,
                        n_clusters=n_clusters,
                        n_noise=n_noise,
                        silhouette_score=sil,
                        calinski_harabasz_score=ch,
                        davies_bouldin_score=db,
                        labels=labels,
                        success=n_clusters > 0
                    )
                    results.append(result)
                    
                    if n_clusters > 1:
                        print(f"      ✅ DBSCAN_{i+1}: {n_clusters} clusters, {n_noise} noise "
                              f"(eps={config['eps']:.3f}, min_samples={config['min_samples']})")
                    
                except Exception as e:
                    print(f"      DBSCAN_{i+1}: Failed - {e}")
                    
        except ImportError:
            print("   Scikit-learn not available for DBSCAN")
            
        return results
    
    def phase2_alternative_methods(self) -> List[ExperimentResult]:
        """Phase 2: Test alternative clustering methods."""
        print("\n🎯 Phase 2: Alternative Clustering Methods")
        results = []
        
        # K-means clustering (force separation)
        print("   Testing K-means variations...")
        try:
            from sklearn.cluster import KMeans
            
            for n_clusters in [2, 3, 4, 5, 6, 7, 8]:
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = clusterer.fit_predict(self.X)
                
                sil, ch, db = self._evaluate_clustering(labels, 'KMeans')
                
                result = ExperimentResult(
                    method=f"KMeans_{n_clusters}",
                    params={'n_clusters': n_clusters},
                    n_clusters=n_clusters,
                    n_noise=0,  # K-means doesn't produce noise
                    silhouette_score=sil,
                    calinski_harabasz_score=ch,
                    davies_bouldin_score=db,
                    labels=labels,
                    success=True,
                    notes=f"Forced {n_clusters} clusters"
                )
                results.append(result)
                
                sil_str = f"{sil:.3f}" if sil is not None else "N/A"
                ch_str = f"{ch:.1f}" if ch is not None else "N/A"
                print(f"      KMeans {n_clusters}: sil={sil_str}, ch={ch_str}")
        except ImportError:
            print("   K-means not available")
        
        # Gaussian Mixture Models
        print("   Testing Gaussian Mixture Models...")
        try:
            from sklearn.mixture import GaussianMixture
            
            for n_components in [2, 3, 4, 5, 6]:
                for covariance_type in ['full', 'tied', 'diag']:
                    try:
                        gmm = GaussianMixture(n_components=n_components, 
                                            covariance_type=covariance_type, 
                                            random_state=42)
                        labels = gmm.fit_predict(self.X)
                        
                        sil, ch, db = self._evaluate_clustering(labels, 'GMM')
                        
                        result = ExperimentResult(
                            method=f"GMM_{n_components}_{covariance_type}",
                            params={'n_components': n_components, 'covariance_type': covariance_type},
                            n_clusters=n_components,
                            n_noise=0,
                            silhouette_score=sil,
                            calinski_harabasz_score=ch,
                            davies_bouldin_score=db,
                            labels=labels,
                            success=True
                        )
                        results.append(result)
                        
                        if sil and sil > 0.3:  # Only print good results
                            print(f"      ✅ GMM {n_components}/{covariance_type}: sil={sil:.3f}")
                            
                    except Exception as e:
                        continue  # Skip failed configs
                        
        except ImportError:
            print("   GMM not available")
        
        # Spectral Clustering
        print("   Testing Spectral Clustering...")
        try:
            from sklearn.cluster import SpectralClustering
            
            for n_clusters in [2, 3, 4, 5]:
                for affinity in ['rbf', 'nearest_neighbors', 'cosine']:
                    try:
                        spec = SpectralClustering(n_clusters=n_clusters, 
                                                affinity=affinity, 
                                                random_state=42)
                        labels = spec.fit_predict(self.X)
                        
                        sil, ch, db = self._evaluate_clustering(labels, 'Spectral')
                        
                        result = ExperimentResult(
                            method=f"Spectral_{n_clusters}_{affinity}",
                            params={'n_clusters': n_clusters, 'affinity': affinity},
                            n_clusters=n_clusters,
                            n_noise=0,
                            silhouette_score=sil,
                            calinski_harabasz_score=ch,
                            davies_bouldin_score=db,
                            labels=labels,
                            success=True
                        )
                        results.append(result)
                        
                        if sil and sil > 0.2:
                            print(f"      ✅ Spectral {n_clusters}/{affinity}: sil={sil:.3f}")
                            
                    except Exception as e:
                        continue
                        
        except ImportError:
            print("   Spectral clustering not available")
        
        # Agglomerative Clustering
        print("   Testing Agglomerative Clustering...")
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            for n_clusters in [2, 3, 4, 5, 6]:
                for linkage in ['ward', 'complete', 'average', 'single']:
                    try:
                        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                        labels = agg.fit_predict(self.X)
                        
                        sil, ch, db = self._evaluate_clustering(labels, 'Agglomerative')
                        
                        result = ExperimentResult(
                            method=f"Agg_{n_clusters}_{linkage}",
                            params={'n_clusters': n_clusters, 'linkage': linkage},
                            n_clusters=n_clusters,
                            n_noise=0,
                            silhouette_score=sil,
                            calinski_harabasz_score=ch,
                            davies_bouldin_score=db,
                            labels=labels,
                            success=True
                        )
                        results.append(result)
                        
                        if sil and sil > 0.2:
                            print(f"      ✅ Agg {n_clusters}/{linkage}: sil={sil:.3f}")
                            
                    except Exception as e:
                        continue
                        
        except ImportError:
            print("   Agglomerative clustering not available")
            
        return results
    
    def phase3_data_preprocessing(self) -> List[ExperimentResult]:
        """Phase 3: Test clustering with different data preprocessing."""
        print("\n🔬 Phase 3: Data Preprocessing Experiments")
        results = []
        
        # PCA preprocessing
        print("   Testing PCA preprocessing...")
        try:
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            
            for n_components in [6, 8, 10]:
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(self.X)
                
                print(f"      PCA {n_components}: explained variance = {pca.explained_variance_ratio_.sum():.3f}")
                
                # Test K-means on PCA data
                for n_clusters in [3, 4, 5]:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(X_pca)
                    
                    # Evaluate using original space for consistency
                    sil, ch, db = self._evaluate_clustering(labels, 'PCA+KMeans')
                    
                    result = ExperimentResult(
                        method=f"PCA{n_components}_KMeans_{n_clusters}",
                        params={'pca_components': n_components, 'n_clusters': n_clusters},
                        n_clusters=n_clusters,
                        n_noise=0,
                        silhouette_score=sil,
                        calinski_harabasz_score=ch,
                        davies_bouldin_score=db,
                        labels=labels,
                        success=True,
                        notes=f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}"
                    )
                    results.append(result)
                    
                    if sil and sil > 0.2:
                        print(f"         ✅ PCA{n_components}+KMeans{n_clusters}: sil={sil:.3f}")
                        
        except ImportError:
            print("   PCA not available")
        
        # Feature engineering: derivatives and ratios
        print("   Testing feature engineering...")
        try:
            # Calculate month-to-month changes (derivatives)
            X_diff = np.diff(self.X, axis=1)
            
            # Test clustering on derivatives
            from sklearn.cluster import KMeans
            for n_clusters in [3, 4, 5]:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(X_diff)
                
                sil, ch, db = self._evaluate_clustering(labels, 'Derivatives+KMeans')
                
                result = ExperimentResult(
                    method=f"Derivatives_KMeans_{n_clusters}",
                    params={'feature_type': 'derivatives', 'n_clusters': n_clusters},
                    n_clusters=n_clusters,
                    n_noise=0,
                    silhouette_score=sil,
                    calinski_harabasz_score=ch,
                    davies_bouldin_score=db,
                    labels=labels,
                    success=True,
                    notes="Using month-to-month production changes"
                )
                results.append(result)
                
                if sil and sil > 0.2:
                    print(f"      ✅ Derivatives+KMeans{n_clusters}: sil={sil:.3f}")
            
            # Test early vs late production ratios
            early_prod = self.X[:, :3].mean(axis=1)  # First 3 months
            late_prod = self.X[:, -3:].mean(axis=1)   # Last 3 months
            ratio_features = np.column_stack([early_prod, late_prod, early_prod/late_prod])
            
            for n_clusters in [3, 4, 5]:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(ratio_features)
                
                sil, ch, db = self._evaluate_clustering(labels, 'Ratios+KMeans')
                
                result = ExperimentResult(
                    method=f"Ratios_KMeans_{n_clusters}",
                    params={'feature_type': 'early_late_ratios', 'n_clusters': n_clusters},
                    n_clusters=n_clusters,
                    n_noise=0,
                    silhouette_score=sil,
                    calinski_harabasz_score=ch,
                    davies_bouldin_score=db,
                    labels=labels,
                    success=True,
                    notes="Using early/late production ratios"
                )
                results.append(result)
                
                if sil and sil > 0.2:
                    print(f"      ✅ Ratios+KMeans{n_clusters}: sil={sil:.3f}")
                    
        except Exception as e:
            print(f"   Feature engineering failed: {e}")
        
        return results
    
    def phase4_parameter_optimization(self, results: List[ExperimentResult]) -> List[ExperimentResult]:
        """Phase 4: Optimize parameters for best performing methods."""
        print("\n🎯 Phase 4: Parameter Optimization")
        
        # Find best performing methods
        successful_results = [r for r in results if r.success and r.n_clusters > 1]
        if not successful_results:
            print("   No successful multi-cluster results to optimize")
            return []
            
        # Sort by silhouette score
        best_methods = sorted([r for r in successful_results if r.silhouette_score], 
                             key=lambda x: x.silhouette_score or 0, reverse=True)[:5]
        
        print(f"   Optimizing top {len(best_methods)} methods:")
        for r in best_methods:
            print(f"      {r.method}: sil={r.silhouette_score:.3f}, {r.n_clusters} clusters")
        
        optimization_results = []
        
        # Fine-tune the best K-means result
        kmeans_results = [r for r in best_methods if 'KMeans' in r.method]
        if kmeans_results:
            best_kmeans = kmeans_results[0]
            print(f"   Fine-tuning K-means (current best: {best_kmeans.n_clusters} clusters)...")
            
            try:
                from sklearn.cluster import KMeans
                
                # Test more granular cluster numbers around the best
                base_k = best_kmeans.n_clusters
                test_ks = list(range(max(2, base_k-2), min(15, base_k+4)))
                
                for k in test_ks:
                    for init in ['k-means++', 'random']:
                        for n_init in [10, 20]:
                            kmeans = KMeans(n_clusters=k, init=init, n_init=n_init, random_state=42)
                            labels = kmeans.fit_predict(self.X)
                            
                            sil, ch, db = self._evaluate_clustering(labels, 'OptimizedKMeans')
                            
                            result = ExperimentResult(
                                method=f"OptKMeans_{k}_{init}_{n_init}",
                                params={'n_clusters': k, 'init': init, 'n_init': n_init},
                                n_clusters=k,
                                n_noise=0,
                                silhouette_score=sil,
                                calinski_harabasz_score=ch,
                                davies_bouldin_score=db,
                                labels=labels,
                                success=True,
                                notes="Optimized K-means parameters"
                            )
                            optimization_results.append(result)
                            
                            if sil and sil > (best_kmeans.silhouette_score or 0) + 0.01:
                                print(f"         ✅ Better K-means found: K={k}, sil={sil:.3f}")
                                
            except Exception as e:
                print(f"   K-means optimization failed: {e}")
        
        return optimization_results
    
    def run_all_experiments(self) -> List[ExperimentResult]:
        """Run all clustering experiments."""
        print("🧪 Starting Comprehensive Clustering Experiments")
        print("=" * 60)
        
        all_results = []
        
        # Run all phases
        all_results.extend(self.phase1_aggressive_current_methods())
        all_results.extend(self.phase2_alternative_methods())
        all_results.extend(self.phase3_data_preprocessing())
        all_results.extend(self.phase4_parameter_optimization(all_results))
        
        return all_results
    
    def generate_recommendations(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Generate actionable recommendations based on experiment results."""
        print("\n🎯 Generating Recommendations")
        print("=" * 40)
        
        # Filter successful results
        successful_results = [r for r in results if r.success and r.n_clusters > 1]
        
        if not successful_results:
            return {
                'status': 'no_solutions',
                'message': 'No clustering method found multiple clusters. Your wells are genuinely very similar.',
                'suggestions': [
                    'Consider expanding your dataset to include different formations or operators',
                    'Try clustering on derived features like decline rates or EUR estimates',
                    'Use time-based clustering (early vs mature wells)',
                    'Accept that these Eagleford wells have very similar performance patterns'
                ]
            }
        
        # Sort by silhouette score
        best_results = sorted([r for r in successful_results if r.silhouette_score], 
                             key=lambda x: x.silhouette_score or 0, reverse=True)
        
        # Find the best result
        best_result = best_results[0]
        
        recommendations = {
            'status': 'solutions_found',
            'best_method': {
                'name': best_result.method,
                'params': best_result.params,
                'n_clusters': best_result.n_clusters,
                'n_noise': best_result.n_noise,
                'silhouette_score': best_result.silhouette_score,
                'notes': best_result.notes
            },
            'top_alternatives': []
        }
        
        # Add top 5 alternatives
        for r in best_results[1:6]:
            recommendations['top_alternatives'].append({
                'name': r.method,
                'n_clusters': r.n_clusters,
                'silhouette_score': r.silhouette_score,
                'params': r.params
            })
        
        # Analysis
        method_types = {}
        for r in successful_results:
            method_type = r.method.split('_')[0]
            if method_type not in method_types:
                method_types[method_type] = []
            method_types[method_type].append(r)
        
        print(f"✅ Found {len(successful_results)} working solutions!")
        print(f"🏆 Best method: {best_result.method}")
        print(f"   - {best_result.n_clusters} clusters")
        print(f"   - Silhouette score: {best_result.silhouette_score:.3f}")
        print(f"   - Parameters: {best_result.params}")
        
        # Default parameter recommendations
        if 'KMeans' in best_result.method:
            recommendations['ui_defaults'] = {
                'clustering_method': 'kmeans',
                'n_clusters': best_result.n_clusters,
                'reason': f'K-means with {best_result.n_clusters} clusters works best for your high-similarity Eagleford data'
            }
        elif 'DBSCAN' in best_result.method:
            recommendations['ui_defaults'] = {
                'clustering_method': 'dbscan',
                'eps': best_result.params.get('eps', 0.3),
                'min_samples': best_result.params.get('min_samples', 3),
                'reason': 'Optimized DBSCAN parameters for high-similarity wells'
            }
        elif 'HDBSCAN' in best_result.method:
            recommendations['ui_defaults'] = {
                'clustering_method': 'hdbscan',
                'min_cluster_size': best_result.params.get('min_cluster_size', 3),
                'min_samples': best_result.params.get('min_samples', 2),
                'reason': 'Ultra-aggressive HDBSCAN for subtle differences'
            }
        
        print(f"\n💡 Recommendations for UI defaults:")
        if 'ui_defaults' in recommendations:
            ui_rec = recommendations['ui_defaults']
            print(f"   - Method: {ui_rec['clustering_method']}")
            print(f"   - Reason: {ui_rec['reason']}")
        
        return recommendations
    
    def save_experiment_results(self, results: List[ExperimentResult], 
                               recommendations: Dict[str, Any], 
                               session_id: str):
        """Save experiment results to a file."""
        # Save to project root, not scripts directory
        output_file = project_root / f"clustering_experiment_{session_id}.json"
        
        # Convert results to serializable format
        serializable_results = []
        for r in results:
            serializable_results.append({
                'method': r.method,
                'params': r.params,
                'n_clusters': int(r.n_clusters),
                'n_noise': int(r.n_noise),
                'silhouette_score': float(r.silhouette_score) if r.silhouette_score is not None else None,
                'calinski_harabasz_score': float(r.calinski_harabasz_score) if r.calinski_harabasz_score is not None else None,
                'davies_bouldin_score': float(r.davies_bouldin_score) if r.davies_bouldin_score is not None else None,
                'success': bool(r.success),
                'notes': r.notes
            })
        
        experiment_data = {
            'session_id': session_id,
            'dataset_info': {
                'n_wells': self.n_wells,
                'n_features': len(self.vector_cols),
                'similarity_stats': self.similarity_stats
            },
            'results': serializable_results,
            'recommendations': recommendations,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        print(f"\n💾 Experiment results saved to: {output_file}")
        return str(output_file)

def main():
    """Main entry point for clustering experiments."""
    
    # Get session ID from command line or use latest
    if len(sys.argv) > 1:
        session_id = sys.argv[1]
    else:
        session_manager = get_session_manager()
        latest = session_manager.get_latest_session()
        if not latest:
            print("❌ No sessions found. Run the Streamlit app first to create session data.")
            return
        session_id = latest['session_id']
        print(f"🔍 Using latest session: {session_id}")
    
    # Load session data
    session_manager = get_session_manager()
    session_data = session_manager.load_session(session_id)
    
    if not session_data:
        print(f"❌ Could not load session: {session_id}")
        return
        
    if 'vectors_df' not in session_data:
        print(f"❌ No vectors_df found in session {session_id}. Run analysis first.")
        return
    
    print(f"✅ Loaded session data: {len(session_data['vectors_df'])} wells")
    
    # Run experiments
    experiment = ClusteringExperiment(session_data)
    results = experiment.run_all_experiments()
    recommendations = experiment.generate_recommendations(results)
    
    # Save results
    output_file = experiment.save_experiment_results(results, recommendations, session_id)
    
    print(f"\n🎉 Clustering experiments completed!")
    print(f"📊 Tested {len(results)} different configurations")
    print(f"💾 Results saved to: {output_file}")
    
    if recommendations['status'] == 'solutions_found':
        best = recommendations['best_method']
        print(f"\n🏆 Best solution: {best['name']}")
        print(f"   - {best['n_clusters']} clusters")
        print(f"   - Silhouette score: {best['silhouette_score']:.3f}")
        print(f"   - Try these parameters in your UI: {best['params']}")
    else:
        print(f"\n❌ {recommendations['message']}")

if __name__ == '__main__':
    main()