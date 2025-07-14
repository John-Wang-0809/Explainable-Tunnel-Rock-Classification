# Survey Analysis Module

This module contains the factor analysis script and associated data for tunnel rock classification survey research.

## Files Structure

```
survey_analysis/
├── factor_analysis.py          # Main factor analysis script (improved version)
├── data/                      # Survey data files
│   └── Raw_Data.csv           # Filtered survey response data
├── outputs/                   # Generated outputs
│   ├── factor_analysis_results.xlsx    # Comprehensive factor analysis results
│   ├── factor_loadings_all.xlsx        # Complete factor loadings matrix
│   ├── factor_loadings_filtered.xlsx   # Filtered factor loadings (>0.4)
│   └── scree_plot.png                  # Scree plot for factor determination
├── documentation/             # Research documentation
│   └── Supplementary Material.docx  # Survey questionnaire content
└── README.md                  # This file
```

## Usage

### Quick Start
```bash
# Navigate to the survey analysis directory
cd survey_analysis/

# Run the factor analysis
python factor_analysis.py
```

# Note: Alternative batch/shell scripts are not included in this version

## Data Description

### Dataset: Raw_Data.csv
This contains the **filtered survey response data** from participants:
- **Source**: Tunnel rock classification trust survey responses
- **Participants**: 204 valid responses after data cleaning
- **Variables**: 
  - **Q1-Q19**: Survey questions (5-point Likert scale)
  - **B1-B5**: Background/demographic variables
  - **Group**: Experimental condition (1=No Explanation, 2=Grad-CAM Explanation)
  - **Random ID**: Participant identifier

### Group Design
- **Group 1**: Participants who evaluated the AI system without Grad-CAM explanations
- **Group 2**: Participants who evaluated the AI system with Grad-CAM explanations

## Analysis Features

### 1. **Exploratory Factor Analysis (EFA)**
   - 4-factor solution using principal axis factoring
   - Varimax rotation for factor interpretation
   - Eigenvalue analysis and scree plot
   - Communalities calculation (h²)

### 2. **Statistical Tests**
   - **Bartlett's Test of Sphericity**: Tests suitability for factor analysis
   - **KMO Test**: Measures sampling adequacy
   - **Normality Tests**: Kolmogorov-Smirnov tests for each factor
   - **Group Differences**: Automatic selection of t-test or Mann-Whitney U

### 3. **Reliability Analysis**
   - **Cronbach's Alpha**: Total scale and individual factors
   - **Factor Scores**: Calculated using regression method
   - **Internal Consistency**: Assessment across all factors

### 4. **Data Presentation**
   - **Formatted Tables**: Statistical results displayed in publication-ready table format
   - **Scree Plot**: Factor determination with Kaiser criterion line (PNG output)
   - **Comprehensive Excel Export**: All results organized in multiple worksheets
   - **Markdown Tables**: Terminal output formatted for easy copying to reports

## Dependencies

### Required Python Packages
```bash
pip install pandas numpy factor_analyzer matplotlib scipy pingouin seaborn
```

### Package Versions (Tested)
- `pandas >= 1.3.0`
- `numpy >= 1.21.0`
- `factor_analyzer >= 0.4.0`
- `matplotlib >= 3.4.0`
- `scipy >= 1.7.0`
- `pingouin >= 0.5.0`
- `seaborn >= 0.11.0`

## Factor Structure

The analysis identifies 4 main factors:
- **RC**: Reliability and Competence (Q1, Q6, Q10, Q13, Q15, Q19)
- **TA**: Trust in Automation (Q5, Q9, Q12, Q14, Q18)
- **UP**: Understanding and Predictability (Q2, Q7, Q11, Q16)
- **FDT**: Familiarity and Developer Trust (Q3, Q4, Q8, Q17)

## Output Files

### Excel Results (`factor_analysis_results.xlsx`)
- **Descriptive Stats and Loadings**: Item statistics with factor loadings
- **Normality Tests**: Kolmogorov-Smirnov test results
- **Group Difference Tests**: Mann-Whitney U test results
- **Initial Eigenvalues**: Complete eigenvalue table with rotation note
- **4-Factor Variance**: Variance explained by each factor after rotation
- **KMO and Bartlett**: Statistical adequacy tests
- **Reliability Analysis**: Cronbach's Alpha for overall scale and individual factors
- **Combined Analysis**: Integrated normality and group difference test table

### Visualizations
- **scree_plot.png**: Factor determination plot with Kaiser criterion

### Factor Loadings
- **factor_loadings_all.xlsx**: Complete factor loading matrix
- **factor_loadings_filtered.xlsx**: Loadings > 0.4 threshold only

## Expected Results

When you run the analysis, you should expect:

### Statistical Adequacy
- **Bartlett's Test**: Significant (p < 0.001) - suitable for factor analysis
- **KMO Test**: > 0.7 - adequate sampling adequacy

### Factor Analysis
- **4 Factors**: Clear factor structure with meaningful loadings
- **Variance Explained**: ~60-80% cumulative variance explained
- **Communalities**: Most items with h² > 0.4

### Group Differences
- **Non-normal Distributions**: All factors show significant deviation from normality
- **Mann-Whitney U Tests**: Used for group comparisons due to non-normality
- **Significant Differences**: Group 2 (with Grad-CAM explanations) shows higher median scores
- **Effect Sizes**: Z-scores indicate moderate to large differences between groups

## Key Features of the Improved Version

### Statistical Rigor
- **Automated Test Selection**: Chooses appropriate statistical tests based on normality assumptions
- **Comprehensive Output**: All results presented in both terminal and Excel formats
- **Publication-Ready Tables**: Formatted markdown tables for direct use in papers

### Enhanced Reporting
- **KMO and Bartlett's Test Table**: Structured presentation of statistical adequacy
- **Variance Explanation Table**: Clear breakdown with rotation method notation  
- **Combined Analysis Table**: Integrated normality and group difference results
- **Cronbach's Alpha Table**: Reliability coefficients for all scales and factors

### Streamlined Visualization
- **Focused Output**: Only essential plot (scree plot) retained
- **Removed Redundancy**: Eliminated cumulative variance and boxplot visualizations
- **High Quality**: 300 DPI PNG output for publication use

## Research Context

This factor analysis supports research on trust in AI-based tunnel rock classification systems, examining how Grad-CAM explanations affect user trust across four key dimensions. The improved analysis provides comprehensive statistical evidence for the effectiveness of explainable AI in engineering applications.

## Troubleshooting

### Common Issues
1. **ModuleNotFoundError**: Install required packages using pip
2. **File not found**: Ensure you're in the correct directory
3. **Permission errors**: Check file permissions for outputs/ folder

### Support
For questions about the analysis or interpretation of results, refer to the documentation in the `documentation/` folder.