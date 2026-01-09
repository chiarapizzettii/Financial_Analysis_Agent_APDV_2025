# Financial Analysis Agent 📊

## TO CHANGE

AI-powered financial data analysis agent using Mistral LLM via Ollama. Query financial data using natural language and get comprehensive analysis with visualizations.

## Features

- 🤖 **Natural Language Interface**: Ask questions in plain English
- 📈 **Automated Visualizations**: Revenue trends, profitability ratios, comparative analysis
- 🔍 **Intelligent Planning**: Mistral LLM creates optimal execution plans
- 📊 **Interactive Dashboard**: Streamlit-based UI with chat interface
- 🛠️ **Extensible Tools**: Modular plotting and calculation functions

## Project Structure

```
.
├── agent/                  # Agent core logic
│   ├── orchestrator.py     # Main orchestrator
│   ├── planner.py          # LLM-based planning
│   ├── executor.py         # Tool execution
├── tools/                  # Analysis tools
│   ├── visualization.py    # Plotly visualizations
│   ├── mapping.py          # Tool mapping
│   ├── reporting.py        # Report creation
│   └── analysis.py         # Financial calculations
├── scripts/               # Setup scripts
│   └── clean_dataset.py   # Data preprocessing
├── data/                  # Data directory
│   ├── processed.csv      # Processed data
│   └── raw.csv            # Raw CSV data
└── app.py                 # Streamlit app

```

## Setup

### Prerequisites

- Python 3.12+
- Ollama
- Mistral model for Ollama

### 1. Install Dependencies

```bash
pip install polars plotly streamlit requests
```

### 2. Install and Setup Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# In another terminal, pull Mistral
ollama pull mistral
```

### 3. Initialize Database

```bash
# Create database from CSV
python scripts/init_db.py

# Normalize into proper tables
python scripts/normalize_db.py
```

### 4. Verify Setup

```bash
# Test the CLI agent
python agent/runner.py
```

## Usage

### Streamlit App (Recommended)

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### CLI Interface

```bash
python agent/runner.py
```

### Example Queries

**Revenue Analysis:**
- "Show me the revenue trend for companies 1, 2, and 3 from 2015 to 2020"
- "How has company 1's revenue grown from 2015 to 2020?"

**Profitability Analysis:**
- "Compare the ROE of companies 1, 2, 3 in 2020"
- "What is the average ROA across all companies in 2020?"
- "Show profitability ratios for company 1 from 2018 to 2020"

**Comparative Analysis:**
- "Compare company 1's ROE against its industry in 2020"
- "Which companies have the highest revenue in 2020?"

**Dashboards:**
- "Show me a dashboard for company 1 in 2020"
- "Create a financial health overview for company 5 in 2019"

## Available Tools

### Plotting Tools

- `plot_revenue_trend()` - Revenue over time
- `plot_profitability()` - ROE, ROA, Net Margin
- `plot_net_income_trend()` - Net income over time
- `plot_comparison()` - Compare companies on a metric
- `plot_industry_benchmark()` - Company vs industry
- `plot_dashboard()` - Comprehensive financial dashboard
- `plot_correlation()` - Metric correlations

### Calculation Tools

- `query_database()` - Execute SQL queries
- `calculate_growth_rate()` - CAGR and growth metrics
- `calculate_aggregate_stats()` - Statistical aggregations

## Configuration

Edit `utils/config.py` or set environment variables:

```bash
export FINANCE_DB_PATH="../data/finance.db"
export OLLAMA_URL="http://localhost:11434"
export OLLAMA_MODEL="mistral"
export AGENT_MAX_RETRIES=2
```

## Architecture

### 1. User Query
User asks a natural language question via Streamlit or CLI

### 2. Planning Phase
`AgentPlanner` uses Mistral to:
- Analyze the query
- Break it into steps
- Select appropriate tools
- Generate execution plan

### 3. Execution Phase
`ToolExecutor` runs the plan:
- Executes SQL queries
- Generates visualizations
- Performs calculations
- Collects results

### 4. Response Generation
`FinancialAgent` synthesizes:
- Tool results into natural language
- Displays plots and tables
- Returns comprehensive response

## Agent Flow

```
User Query
    ↓
[AgentPlanner]
    ↓
Execution Plan
    ↓
[ToolExecutor]
    ↓
Tool Results
    ↓
[Response Generator]
    ↓
Final Answer + Visualizations
```

## Extending the Agent

### Add New Plotting Function

1. Add function to `tools/plotting.py`
2. Define tool schema in `agent/schema.py`
3. Add routing in `agent/executor.py`

### Add New Calculation

1. Add function to `tools/calculations.py`
2. Define tool schema in `agent/schema.py`
3. Add routing in `agent/executor.py`

## Troubleshooting

### "Ollama is not running"
```bash
# Start Ollama
ollama serve
```

### "Model 'mistral' not found"
```bash
# Pull the model
ollama pull mistral
```

### "Database not found"
```bash
# Check database path
ls -la data/finance.db

# Reinitialize if needed
python scripts/init_db.py
python scripts/normalize_db.py
```

### "No module named 'polars'"
```bash
# Install dependencies
pip install polars plotly streamlit requests
```

## Performance Tips

- **Limit company_ids**: Query fewer companies for faster responses
- **Use specific years**: Narrow date ranges reduce processing time
- **Cache results**: The database has indexes on key columns

## Known Limitations

- No support for localStorage (runs in standard Python environment)
- Requires local Ollama installation
- Database must fit in memory for Polars operations
- Single-user operation (no concurrent request handling)

## Future Enhancements

- [ ] Add more financial metrics (P/E ratio, debt ratios, etc.)
- [ ] Support for custom metric definitions
- [ ] Export reports to PDF/Excel
- [ ] Multi-year comparative analysis
- [ ] Industry sector analysis
- [ ] Forecasting and trend prediction
- [ ] API endpoint for programmatic access

## License

MIT License - feel free to use and modify for your needs

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section above
- Review example queries in the sidebar
- Inspect agent execution details in Streamlit
