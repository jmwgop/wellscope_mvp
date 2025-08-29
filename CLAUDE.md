# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
```bash
# Install the project
pip install -e .
pip install -e ".[dev]"  # With development dependencies

# Run tests (164 tests across 25 files)
pytest                   # All tests
pytest tests/app/        # Frontend tests only (131 tests)
pytest wellscope_mvp/    # Backend pipeline tests only

# Run the Streamlit application
streamlit run main.py    # Opens at http://localhost:8501
```

### Testing Structure
- Backend tests: `tests/` (33 tests for core pipeline)
- Frontend tests: `tests/app/` (131 tests for Streamlit components)
- Test fixtures: Small CSV files in `tests/fixtures/`
- Large test data: `fixtures_big/` directory

## Architecture

### High-Level Structure
This is an oil & gas well data analysis platform with two main parts:

1. **Backend Pipeline** (`wellscope_mvp/`): Pure Python data processing
   - Data layer: CSV loading with schema validation
   - Pipeline layer: Join, filter, vector building, clustering, UMAP projection
   - Designed for Enverus/DrillingInfo format CSVs (Well Headers + Monthly Production)

2. **Streamlit Frontend** (`app/`): Interactive web UI with 6-layer architecture
   - Components, Services, Utils, State, Config, Pages
   - Complete workflow from upload to ML-powered clustering visualization
   - Smart Analysis Configuration system with data-driven recommendations
   - Immediate insights and simplified controls for non-ML users

### Key Data Flow
```
CSV Upload → Schema Validation → Join (API14 normalization) → 
Filter (formations, dates, production) → Smart Analysis (immediate insights) → 
Vector Building (production curves) → ML Clustering (production-optimized) → 
UMAP Projection → Similarity Scoring → Interactive Visualization
```

### Core Schemas
- **Headers**: 21 fields (API14, coordinates, formations, completion data)
- **Monthly**: 7 fields (API_UWI, dates, oil/gas/water volumes)
- Clean separation: no field duplication between datasets

### ML Pipeline Components
- **Smart Analysis Configuration**: Immediate data analysis with user-friendly recommendations
- **Production Optimization**: Automatic DBSCAN parameter optimization for high-similarity oil & gas data
- **Vector Building**: Production curve shapes with peak normalization or decline rates
- **Clustering**: HDBSCAN preferred, DBSCAN fallback, with production-aware parameter selection
- **Mature Well Analysis**: Specialized clustering for wells with 12+ months production
- **Similarity Scoring**: Euclidean distance-based well similarity analysis

### Frontend Architecture
- **Smart Analysis System**: Data-driven configuration with immediate insights (30-second setup)
- **Production Optimization UI**: Transparent messaging about why aggressive parameters are used
- **Session Management**: Persistent state across interactions with fallback for non-Streamlit environments
- **Caching**: Streamlit-aware caching of expensive pipeline operations
- **Component System**: Reusable UI components (upload, filter, smart analysis, plots, tables)
- **Intelligent UX**: Data-aware recommendations, real-time validation, simplified controls

### Testing Strategy
- **Comprehensive Coverage**: 164 tests total with both unit and integration tests
- **Mock Testing**: All components work without Streamlit environment
- **Fixture-based**: Uses real CSV samples for realistic testing
- **Validation Testing**: Extensive input validation and error handling coverage

### Configuration
- Backend config via dataclasses (`FilterConfig`, `VectorConfig`, `ClusterConfig`)
- Frontend defaults in `app/config/ui_defaults.py`
- Streamlit configuration in `.streamlit/config.toml` (if present)

### File Upload Workflow
The app expects two CSV files:
1. Well Headers CSV (completion metadata)
2. Producing Entity Monthly Production CSV (time-series data)
Both are joined on normalized API14 identifiers with comprehensive statistics tracking.

## Key Implementation Notes

### Error Handling
- Comprehensive validation at each pipeline stage
- Graceful fallbacks (UTF-8/Latin-1 encoding, NaN handling)
- User-friendly error messages with actionable guidance

### Performance
- Streamlit caching for expensive operations
- Memory-efficient DataFrame operations
- Intelligent parameter selection to avoid common pitfalls

### Data Quality
- Schema validation on CSV load
- Join statistics and quality metrics
- Production history validation (months-produced calculations)
- Well maturity analysis for optimal vector lengths

### Session Persistence
- Auto-save at major workflow steps (upload, filter, analysis completion)
- Session recovery UI with auto-detection of recent sessions
- Complete analysis state preservation across browser refreshes/restarts
- Command-line session management utilities
- Sessions stored in `.wellscope_sessions/` directory with CSV + JSON format
- Automatic cleanup of old sessions (configurable retention)

#### Session Management Commands
```bash
# List all sessions
python -m app.utils.session_utils list --details

# Get session information
python -m app.utils.session_utils info <session_id>

# Clean up old sessions
python -m app.utils.session_utils cleanup --dry-run

# Delete specific session
python -m app.utils.session_utils delete <session_id>

# Export session to directory
python -m app.utils.session_utils export <session_id> --output-dir <path>
```

#### Clustering Experiment Scripts
```bash
# Test different clustering approaches on session data
python scripts/experiment_clustering.py [session_id]

# Analyze clustering experiment results  
python scripts/analyze_experiment_results.py

# Both scripts save results to project root and provide actionable recommendations
# for improving clustering parameters when wells have high similarity (90%+)
```

#### Session Recovery
- Auto-recovery banner appears when recent sessions are available
- Manual session recovery panel in main workflow
- Session management controls in sidebar
- Complete data and analysis state restoration

## Smart Analysis Configuration System

### Overview
The Smart Analysis Configuration system replaces complex ML parameter tuning with data-driven recommendations, transforming Step 3 from a 5+ minute technical configuration into a 30-second guided experience.

### Key Components

**Core Files:**
- `app/components/smart_analysis_config.py` - Main smart configuration UI component
- `app/utils/data_analyzer.py` - Immediate data analysis for recommendations
- `app/utils/smart_recommendations.py` - User-friendly message generation and config conversion

### User Experience Flow
1. **Immediate Analysis**: After filtering, system instantly analyzes data characteristics
2. **Smart Insights**: "Found 847 similar oil wells with 85% similarity"
3. **Clear Recommendations**: "We recommend 8 clusters with 50-150 wells each"
4. **Simple Controls**: 3-4 sliders to adjust preferences (cluster count, group size, history length)
5. **Real-time Preview**: "Will create ~8 groups with 70-120 wells each"
6. **Transparent Optimization**: Shows why production-optimized parameters are used

### Production Optimization Integration
- **Automatic Detection**: Identifies oil & gas production patterns from data characteristics
- **Optimal Parameters**: Applies experimental findings (eps=0.05-0.08 for high similarity)
- **User Transparency**: Explains why aggressive parameters are recommended
- **Success Rate**: Transforms 0% to 100% success rate on production datasets

### Technical Architecture
- **Data Analysis**: Extracts formation diversity, completion timeline, well maturity without vectors
- **Smart Recommendations**: Converts technical analysis to user-friendly insights
- **Config Translation**: Converts simple user preferences to optimal technical parameters
- **Backend Compatibility**: Maintains full compatibility with existing pipeline architecture