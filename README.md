# WellScope MVP

Complete oil & gas well data processing and ML analysis platform - from raw CSVs to production curve clusters and interactive visualizations. Streamlined schemas with intelligent joining, filtering, clustering, and similarity analysis.

## Features

### **Data Foundation**
- **Headers Loader**: 21 core well completion fields (API, coordinates, formations, completion data)
- **Monthly Loader**: 7 time-series production fields (volumes, dates, well counts)
- **Clean Architecture**: No field duplication between datasets, optimized for joins
- **Robust Processing**: UTF-8/Latin-1 encoding fallback, type coercion, schema validation

### **Pipeline Processing**
- **Smart Joining**: Intelligent API14 normalization with comprehensive join statistics
- **Advanced Filtering**: Multi-criteria filtering (formations, operators, completion dates, production history)
- **Smart Analysis Configuration**: Data-driven recommendations with immediate insights for non-ML users
- **Production Optimization**: Automatic DBSCAN parameter optimization for high-similarity oil & gas data
- **ML Feature Engineering**: Production curve shape vectors with peak normalization and decline rate analysis
- **Intelligent Clustering**: Production-aware parameter selection with experimental optimization results

### **Quality & Testing**
- **Full Test Coverage**: 164 comprehensive tests across 25 test files including pipeline integration
- **Production Statistics**: Join quality metrics, filter coverage, data quality assessment
- **Flexible Configuration**: Dataclass-driven filtering and vector building

## Installation

```bash
pip install -e .
# or for development
pip install -e ".[dev]"
```

## Quick Start - Complete Pipeline with Smart Analysis

```python
from wellscope_mvp.data import load_headers_csv, load_monthly_csv
from wellscope_mvp.pipeline import join_headers_and_monthly, FilterConfig, apply_filters, build_shape_vectors, VectorConfig
from app.utils.data_analyzer import analyze_filtered_data
from app.utils.smart_recommendations import generate_smart_recommendations

# 1. Load raw data
headers_df, headers_meta = load_headers_csv("Well Headers.CSV")
monthly_df, monthly_meta = load_monthly_csv("Producing Entity Monthly Production.CSV")
print(f"Loaded {headers_meta['n_rows']} wells, {monthly_meta['n_rows']} monthly records")

# 2. Intelligent joining with statistics
joined_df, join_stats = join_headers_and_monthly(headers_df, monthly_df, how="inner")
print(f"Joined {join_stats['matched_api14']} wells ({join_stats['joined_rows']} records)")

# 3. Apply business filters
filter_config = FilterConfig(
    formations=["EAGLEFORD"],
    completion_year_range=(2020, 2024),
    min_months_produced=12,
    lateral_ft_range=(5000, 15000)
)
filter_result = apply_filters(joined_df, filter_config)
filtered_df = filter_result["filtered"]
print(f"Filtered to {len(filtered_df)} records ({filter_result['stats']['kept_fraction']:.2%} kept)")

# 4. Smart Analysis - Immediate insights for data-driven configuration
data_analysis = analyze_filtered_data(filtered_df, filter_config.__dict__)
recommendations = generate_smart_recommendations(data_analysis)
print(f"Found {data_analysis['n_wells']} wells with {data_analysis['similarity_mean']:.1%} similarity")
print(f"Recommended: {recommendations['recommended_clusters']} clusters with production optimization")

# 5. Generate ML-ready production shape vectors (now with smart defaults)
vector_config = VectorConfig(
    months=recommendations['optimal_months'], 
    stream=recommendations['production_stream'], 
    normalize="q_over_qmax"
)
vectors_df, vector_meta = build_shape_vectors(filtered_df, vector_config)
print(f"Generated {vector_meta['n_wells']} well vectors ({vector_meta['n_features']} features each)")
```

## Architecture

### **3-Layer Design**

#### **1. Data Layer** (`wellscope_mvp.data`)
Raw CSV ingestion with schema validation:
- **Headers**: Well completion metadata (21 fields)
- **Monthly**: Production time-series (7 fields)
- Encoding fallback, type coercion, normalization

#### **2. Pipeline Layer** (`wellscope_mvp.pipeline`)  
Data processing and preparation:
- **Join**: Smart API14 matching with statistics
- **Filter**: Multi-criteria business rule filtering  
- **Vectors**: ML-ready feature generation

#### **3. Analysis Layer** (Future)
Ready for ML models, clustering, forecasting

## Data Schemas

### Headers Schema (21 fields)
Core well completion and drilling data:
- **Identity**: API14, Operator, County  
- **Location**: Surface/Bottom hole coordinates
- **Geology**: DI Play/Subplay, Target Formation, Producing Reservoir
- **Drilling**: Drill Type, depths, lateral lengths, perforations
- **Timeline**: Spud, Completion, First Prod dates

### Monthly Schema (7 fields)  
Pure time-series production metrics:
- **Key**: API/UWI (joins to headers API14)
- **Time**: Monthly Production Date, Producing Month Number
- **Volumes**: Monthly Oil, Gas, Water
- **Context**: Well Count

## API Reference

### Data Loading

#### `load_headers_csv(path) -> (DataFrame, dict)`
Load well headers with schema validation and type coercion.

**Returns:**
- `DataFrame`: Headers data with normalized API14
- `dict`: Metadata (n_rows, path)

#### `load_monthly_csv(path) -> (DataFrame, dict)`  
Load monthly production with minimal time-series schema.

**Returns:**
- `DataFrame`: Production data with API_UWI_norm column
- `dict`: Metadata (n_rows, path)

### Pipeline Processing

#### `join_headers_and_monthly(headers_df, monthly_df, how="inner") -> (DataFrame, dict)`
Intelligent joining with API14 normalization.

**Parameters:**
- `how`: Join strategy ("inner", "left", "outer")

**Returns:**
- `DataFrame`: Joined dataset with normalized keys
- `dict`: Statistics (matched wells, coverage metrics)

#### `apply_filters(joined_df, FilterConfig) -> dict`
Apply business filters to joined data.

**FilterConfig options:**
- `formations`: List of target formations
- `completion_year_range`: (min_year, max_year) tuple
- `lateral_ft_range`: (min_ft, max_ft) lateral length
- `min_months_produced`: Minimum producing months
- `operators`: List of operator names

**Returns:**
- `dict`: {'filtered': DataFrame, 'mask': Series, 'stats': dict}

#### `build_shape_vectors(joined_df, VectorConfig) -> (DataFrame, dict)`
Generate ML-ready production shape vectors.

**VectorConfig options:**
- `months`: Vector length (default: 24)
- `stream`: "oil", "gas", "water", or "boe" (barrel of oil equivalent, gas/6.0)
- `normalize`: "q_over_qmax" (peak-normalized 0-1) or "pct_decline" (month-over-month rates)
- `boe_gas_factor`: Gas-to-oil conversion factor for BOE (default: 6.0)

**Returns:**
- `DataFrame`: Well vectors [api, v00, v01, ..., v23]
- `dict`: Metadata (n_wells, n_features)

## Development

```bash
# Run tests
pytest

# Install dev dependencies
pip install -e ".[dev]"

# Project structure
wellscope_mvp/
├── data/           # CSV loaders with schema validation
├── schema/         # Field definitions & validation logic
├── pipeline/       # Join, filter, and ML prep tools
└── tests/          # Comprehensive test suite (164 tests across 25 files)
```

## Smart Analysis Features

### **Data-Driven Configuration**
- **Immediate Insights**: Analyzes filtered data to show similarity, production type, confidence
- **Smart Recommendations**: "Found 847 similar oil wells → Recommend 8 clusters"  
- **Simple Controls**: 3-4 intuitive sliders instead of 10+ technical parameters
- **Real-time Preview**: Shows expected cluster outcomes before running analysis

### **Production Optimization**
- **Automatic Detection**: Identifies oil & gas production patterns from data characteristics
- **Proven Parameters**: Applies experimental findings (eps=0.05-0.08 for high-similarity data)
- **Success Rate**: Transforms 0% to 100% success rate on production datasets
- **Transparent UI**: Explains why aggressive parameters are recommended

### **User Experience**
- **30-Second Setup**: Configuration takes seconds instead of minutes
- **Non-ML Friendly**: Clear insights and recommendations for non-technical users
- **Progressive Disclosure**: Simple by default, advanced options available when needed

## Use Cases

### **Production Analysis**
- Production decline type curves and EUR modeling with smart parameter selection
- Well performance comparison via shape vector similarity with production optimization
- Type curve analysis by formation/completion parameters with intelligent clustering

### **Asset Screening** 
- Filter wells by geological, completion, or performance criteria with immediate insights
- Portfolio analysis and benchmarking with data-driven recommendations
- Acquisition target identification with smart similarity analysis

### **Machine Learning**
- Production curve clustering by shape similarity with automatic parameter optimization
- Well performance prediction and EUR forecasting with smart feature engineering
- Completion parameter optimization models with production-aware clustering
- Automated decline curve analysis with intelligent parameter selection

## Data Sources

Designed for Enverus/DrillingInfo format CSVs:
- Well Headers CSV (completion data)  
- Producing Entity Monthly Production CSV (time-series)

## License

MIT