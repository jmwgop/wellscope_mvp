# WellScope MVP - Streamlit Frontend

Interactive web application for oil & gas well data analysis - complete UI workflow from CSV upload to ML-powered clustering visualization. Built with Streamlit and modular component architecture for scalable, maintainable data science applications.

## Features

### **Interactive Workflow**
- **Guided Process**: 4-step streamlined workflow from data upload to exportable results
- **Smart Analysis Configuration**: Data-driven recommendations replace complex parameter tuning
- **Real-time Validation**: File validation, data quality checks, intelligent configuration
- **Progress Tracking**: Visual progress indicators and status updates throughout analysis
- **Error Handling**: Comprehensive error messages with actionable guidance

### **Smart Components**
- **Upload Panel**: Drag-drop file upload with CSV validation and data preview
- **Filter Panel**: Dynamic filtering by formation, operator, completion date, production history
- **Smart Analysis Config**: Data-driven configuration with immediate insights and simple controls
- **Production Optimization**: Automatic parameter optimization with transparent user messaging
- **Visualization Engine**: Interactive Plotly charts with zoom, pan, and selection capabilities
- **Data Tables**: Paginated tables with sorting, filtering, and export functionality

### **Performance & Reliability**
- **Streamlit Caching**: Intelligent caching of pipeline results for faster re-runs
- **Session Management**: Persistent state across user interactions
- **Fallback Architecture**: Graceful degradation when Streamlit unavailable (testing/CI)
- **Memory Optimization**: Efficient DataFrame handling for large datasets

## Quick Start

### **Running the Application**
```bash
# From project root
streamlit run main.py

# Application will open in browser at http://localhost:8501
```

### **Example Workflow**
1. **Upload Data**: Drag CSV files (Well Headers + Monthly Production)
2. **Apply Filters**: Select formations, date ranges, production criteria with real-time preview
3. **Smart Analysis**: Get immediate insights ("Found 847 similar oil wells") and simple configuration
4. **Run Analysis**: Execute optimized ML pipeline and explore interactive results

### **Sample Screenshots**
```
┌─────────────────────────────────────────────────┐
│ 🛢️ WellScope MVP - Well Similarity Analysis    │
├─────────────────────────────────────────────────┤
│ 📁 Step 1: Upload Data                         │
│ ┌─────────────┐  ┌─────────────┐              │
│ │ Well Headers│  │ Monthly Prod│              │
│ │ [Drop CSV]  │  │ [Drop CSV]  │              │
│ └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────┤
│ 🔍 Step 2: Filter Data                         │
│ Formations: [EAGLEFORD ✓] [AUSTIN CHALK ✓]    │
│ Completion Years: [2020] ──── [2024]          │
│ Production Months: [≥ 12 months]               │
├─────────────────────────────────────────────────┤
│ 🎯 Step 3: Smart Analysis                      │
│ 📊 Found: 847 similar oil wells (85% similar)  │
│ 💡 Recommend: 8 clusters (50-150 wells each)   │
│ Well Groups: [2] ────●──── [15]               │
│ Production History: [6] ──●── [36] months      │
│ 🛢️ Production optimization enabled              │
└─────────────────────────────────────────────────┘
```

## Architecture

### **6-Layer Design**

#### **1. Components Layer** (`app/components/`)
Reusable UI components with Streamlit integration:
- **`upload_panel.py`**: File upload, validation, and preview
- **`filter_panel.py`**: Dynamic data filtering interface
- **`smart_analysis_config.py`**: Data-driven configuration with immediate insights (NEW)
- **`cluster_controls.py`**: Advanced ML parameter configuration panels (legacy)
- **`plots.py`**: Interactive Plotly visualization components
- **`tables.py`**: Paginated data tables with export functionality

#### **2. Services Layer** (`app/services/`)
Business logic and backend integration:
- **`pipeline_driver.py`**: ML pipeline orchestration and execution
- **`io.py`**: File I/O operations and data loading services
- **`caching.py`**: Streamlit-aware caching with fallback support

#### **3. Utils Layer** (`app/utils/`)
Shared utilities and helper functions:
- **`data_analyzer.py`**: Immediate data analysis for smart recommendations (NEW)
- **`smart_recommendations.py`**: User-friendly message generation and config conversion (NEW)
- **`formatting.py`**: Data formatting for display (numbers, dates, units)
- **`validation.py`**: Input validation and error handling
- **`guards.py`**: Safety checks and data quality validation
- **`clustering_intelligence.py`**: Data-aware parameter selection and intelligent cluster suggestions
- **`mature_well_clustering.py`**: Well maturity analysis and specialized clustering strategies

#### **4. State Layer** (`app/state/`)
Session and state management:
- **`session.py`**: Streamlit session state wrapper with type safety

#### **5. Config Layer** (`app/config/`)
Configuration and defaults:
- **`ui_defaults.py`**: UI parameter defaults and configuration

#### **6. Pages Layer** (`app/pages/`)
Complete page implementations:
- **`page_upload_analyze.py`**: Main analysis workflow page

### **Data Flow Architecture**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Upload      │───▶│ Session     │───▶│ Pipeline    │
│ Components  │    │ State       │    │ Services    │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Validation  │    │ Caching     │    │ Backend     │
│ Utils       │    │ Layer       │    │ Pipeline    │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Display     │◀───│ Results     │◀───│ ML          │
│ Components  │    │ Processing  │    │ Analysis    │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Component API Reference

### Upload Panel
**`render_upload_panel() -> Dict[str, Any]`**

Interactive file upload with validation and preview.

**Returns:**
```python
{
    'headers_df': pd.DataFrame,    # Well headers data
    'monthly_df': pd.DataFrame,    # Monthly production data
    'headers_meta': dict,          # File metadata
    'monthly_meta': dict,          # File metadata  
    'upload_status': str,          # 'pending', 'success', 'error'
    'errors': List[str],           # Validation errors
    'files_uploaded': bool         # Upload completion flag
}
```

### Filter Panel
**`render_filter_panel(joined_df: pd.DataFrame) -> Dict[str, Any]`**

Dynamic filtering interface with real-time preview.

**Parameters:**
- `joined_df`: Combined well headers and production data

**Returns:**
- `dict`: Filter configuration for pipeline

### Smart Analysis Configuration (NEW)
**`render_smart_analysis_config(filtered_df: pd.DataFrame, filters_cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]`**

Data-driven analysis configuration with immediate insights and simple controls.

**Features:**
- **Immediate Analysis**: Analyzes filtered data to show similarity, production type, confidence
- **Smart Insights**: "Found 847 similar oil wells with 85% similarity"
- **Clear Recommendations**: "We recommend 8 clusters with 50-150 wells each"
- **Simple Controls**: 3-4 intuitive sliders (cluster count, group size, history length)
- **Real-time Preview**: "Will create ~8 groups with 70-120 wells each"
- **Production Optimization**: Automatic parameter optimization with transparent messaging

**Returns:**
- `smart_config`: Complete configuration including user preferences and technical configs
- `is_valid`: Configuration validity (always true - smart mode handles edge cases)

### Cluster Controls (Legacy)
**`render_all_controls() -> Tuple[Dict[str, Any], bool, List[str]]`**

Advanced ML parameter configuration interface for expert users.

**Features:**
- Data-aware parameter recommendations based on dataset characteristics
- Well maturity analysis for vector length optimization
- Formation diversity scoring for cluster size suggestions
- Real-time validation with user-friendly error messages

**Returns:**
- `configs`: Combined configuration objects (vector, cluster, projection)
- `is_valid`: Overall validation status
- `errors`: List of configuration issues

### Interactive Plots
**`render_interactive_plots(pipeline_results: Dict[str, Any]) -> None`**

Plotly-based visualization suite.

**Features:**
- 2D/3D scatter plots with cluster coloring
- Production curve overlays by cluster
- Similarity score distributions
- Interactive selection and filtering

### Data Tables
**`render_well_data_table(joined_df: pd.DataFrame, ...) -> None`**

Paginated, sortable data display.

**Features:**
- Server-side pagination for large datasets
- Column sorting and filtering
- CSV export functionality
- Cluster assignment integration

## Configuration

### UI Defaults (`app/config/ui_defaults.py`)
```python
# Vector building
vector_months: int = 24
vector_stream: str = "oil"  # 'oil', 'gas', 'water', 'boe'
vector_normalize: str = "q_over_qmax"

# Clustering  
min_cluster_size: int = 20
use_hdbscan: bool = True
clustering_metric: str = "euclidean"

# UMAP projection
umap_n_components: int = 2
umap_n_neighbors: int = 15
umap_min_dist: float = 0.1

# UI behavior
max_file_size_mb: int = 100
default_page_size: int = 20
similarity_threshold: float = 0.7
```

### Streamlit Configuration
```toml
# .streamlit/config.toml
[server]
maxUploadSize = 100
enableCORS = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
```

## Testing

### **Running App Tests**
```bash
# Run all app tests (131 test functions)
pytest tests/app/ -v

# Run specific component tests
pytest tests/app/test_upload_panel.py -v
pytest tests/app/test_filter_panel.py -v
pytest tests/app/test_cluster_controls.py -v

# Run with coverage
pytest tests/app/ --cov=app --cov-report=html
```

### **Test Structure**
- **Component Tests**: Each UI component has comprehensive test coverage
- **Integration Tests**: End-to-end workflow testing
- **Mock Testing**: Tests work without Streamlit environment
- **Validation Tests**: Input validation and error handling

### **Test Coverage**
```
App Test Functions:  131 tests
App Test Files:       14 files
Total Project Tests: 164 tests (25 files)
```

## Development Guide

### **Adding New Components**
1. Create component file in `app/components/`
2. Follow the pattern:
```python
def render_my_component() -> ReturnType:
    """Component description."""
    if not STREAMLIT_AVAILABLE:
        return _mock_my_component()
    
    # Streamlit UI code here
    st.subheader("Component Title")
    # ... implementation
    
def _mock_my_component() -> ReturnType:
    """Mock for testing."""
    return default_return_value
```
3. Add comprehensive tests in `tests/app/test_my_component.py`
4. Import and integrate in main page

### **Extending Pipeline Integration**
1. Add new pipeline functions to `app/services/pipeline_driver.py`
2. Update configuration classes in `app/config/ui_defaults.py`
3. Add validation rules in `app/utils/validation.py`
4. Create corresponding UI controls in components
5. Add intelligent parameter selection logic to `app/utils/clustering_intelligence.py`
6. Update well maturity analysis in `app/utils/mature_well_clustering.py`

### **Styling Guidelines**
- Use Streamlit's built-in styling (columns, expanders, sidebars)
- Consistent emoji usage for visual hierarchy (📁 🔍 ⚙️ 📊)
- Color coding: Success (green), Warning (orange), Error (red)
- Responsive design with `st.columns()` for different screen sizes

## Deployment

### **Streamlit Cloud**
```python
# requirements.txt additions for Streamlit Cloud
streamlit>=1.28.0
plotly>=5.15.0
pandas>=1.5.3
numpy>=1.24.3
```

### **Docker Deployment**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### **Environment Variables**
```bash
# Optional configurations
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## Performance Optimization

### **Caching Strategy**
- Pipeline results cached based on input data fingerprint
- Streamlit `@st.cache_data` for expensive computations
- Fallback caching for non-Streamlit environments

### **Memory Management**
- Efficient DataFrame operations with column selection
- Pagination for large data tables
- Session state cleanup for large objects

### **User Experience**
- Progress bars for long-running operations
- Real-time validation feedback
- Responsive design for different screen sizes
- Keyboard shortcuts and accessibility features

## Support

For frontend-specific issues:
- Component rendering problems
- Streamlit configuration issues  
- UI/UX improvements
- Performance optimization

See main project README for backend pipeline support.