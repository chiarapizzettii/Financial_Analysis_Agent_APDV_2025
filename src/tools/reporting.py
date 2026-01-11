# src/tools/reporting.py

"""
Report generation tools for financial analysis
----------------------------------------------

Generates comprehensive PDF reports from execution states.
"""

import datetime
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import polars as pl
from fpdf import FPDF

# ============================================================
# Report Generation Classes
# ============================================================


class CompanyReport(FPDF):
    """Enhanced PDF report with improved styling."""

    def __init__(self, company_name: str = "Financial Analysis"):
        super().__init__()
        self.company_name = company_name
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        """Custom header with styling."""
        # Colored header bar
        self.set_fill_color(41, 128, 185)
        self.rect(0, 0, 210, 25, "F")

        # Title
        self.set_y(8)
        self.set_font("Arial", "B", 18)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "Financial Analysis Report", 0, 1, "C")

        if self.company_name:
            self.set_font("Arial", "I", 10)
            self.cell(0, 5, self.company_name, 0, 1, "C")

        self.set_text_color(0, 0, 0)
        self.ln(5)

    def footer(self):
        """Custom footer with page number."""
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")
        self.set_text_color(0, 0, 0)

    def add_section_title(self, title: str):
        """Add a styled section title."""
        self.ln(5)
        self.set_font("Arial", "B", 14)
        self.set_fill_color(230, 240, 250)
        self.cell(0, 10, title, 0, 1, "L", True)
        self.ln(3)

    def add_table(
        self,
        headers: List[str],
        data: List[List[str]],
        col_widths: Optional[List[int]] = None,
    ):
        """Add a formatted table to the report."""
        if not col_widths:
            available_width = 190
            col_widths = [available_width / len(headers)] * len(headers)

        # Table header
        self.set_font("Arial", "B", 11)
        self.set_fill_color(52, 152, 219)
        self.set_text_color(255, 255, 255)

        for i, header in enumerate(headers):
            self.cell(col_widths[i], 10, str(header)[:20], 1, 0, "C", True)
        self.ln()

        # Table rows
        self.set_font("Arial", "", 10)
        self.set_text_color(0, 0, 0)
        fill = False

        for row in data:
            for i, cell in enumerate(row):
                if fill:
                    self.set_fill_color(240, 248, 255)
                else:
                    self.set_fill_color(255, 255, 255)
                self.cell(col_widths[i], 8, str(cell)[:20], 1, 0, "C", True)
            self.ln()
            fill = not fill

        self.ln(5)

    def add_key_metrics(self, metrics: Dict[str, str]):
        """Add key metrics in card format."""
        self.set_font("Arial", "B", 11)
        card_width = 90
        card_height = 25
        x_start = 10
        y_start = self.get_y()

        colors = [
            (46, 204, 113),  # Green
            (52, 152, 219),  # Blue
            (155, 89, 182),  # Purple
            (241, 196, 15),  # Yellow
        ]

        for idx, (metric, value) in enumerate(metrics.items()):
            row = idx // 2
            col = idx % 2

            x = x_start + col * (card_width + 10)
            y = y_start + row * (card_height + 5)

            # Draw card background
            self.set_fill_color(*colors[idx % len(colors)])
            self.rect(x, y, card_width, card_height, "F")

            # Metric name
            self.set_xy(x + 5, y + 5)
            self.set_font("Arial", "B", 10)
            self.set_text_color(255, 255, 255)
            self.cell(card_width - 10, 8, metric[:30], 0, 1)

            # Metric value
            self.set_xy(x + 5, y + 13)
            self.set_font("Arial", "B", 14)
            self.cell(card_width - 10, 8, str(value)[:20], 0, 1)

        self.set_text_color(0, 0, 0)
        self.set_y(y_start + ((len(metrics) + 1) // 2) * (card_height + 5) + 10)


# ============================================================
# Main Report Generation Function
# ============================================================


def generate_report_from_state(
    state,  # ExecutionState object
    user_query: str = "",
    output_dir: str = "outputs",
    custom_filename: Optional[str] = None,
) -> str:
    """
    Generate a PDF report from ExecutionState.

    Args:
        state: ExecutionState object with analysis results
        user_query: The original user query
        output_dir: Directory to save the report
        custom_filename: Custom filename (without extension)

    Returns:
        Path to the generated PDF file
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate filename
    if custom_filename:
        filename = (
            custom_filename
            if custom_filename.endswith(".pdf")
            else f"{custom_filename}.pdf"
        )
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"financial_report_{timestamp}.pdf"

    filepath = output_path / filename

    # Create PDF
    pdf = CompanyReport(company_name="Sequential Analysis Report")
    pdf.add_page()

    # Date and query
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(100, 100, 100)
    date_str = datetime.date.today().strftime("%d/%m/%Y")
    pdf.cell(0, 8, f"Generated on: {date_str}", 0, 1)
    pdf.set_text_color(0, 0, 0)

    if user_query:
        pdf.ln(3)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "Query:", 0, 1)
        pdf.set_font("Arial", "I", 10)
        pdf.multi_cell(0, 6, user_query)

    pdf.ln(5)

    # Key metrics from state
    summary = state.get_summary()
    key_metrics = {
        "Data Points": f"{summary['dataframe_shape'][0]:,}",
        "Metrics": str(summary["dataframe_shape"][1]),
        "Visualizations": str(summary["num_visualizations"]),
        "Tables": str(summary["num_tables"]),
    }

    pdf.add_section_title("Executive Summary")
    pdf.add_key_metrics(key_metrics)

    # Add summary tables
    if state.tables:
        pdf.add_section_title("Summary Statistics")

        for table_info in state.tables:
            title = table_info.get("title", "Data Table")
            data = table_info["data"]

            if isinstance(data, pl.DataFrame):
                # Convert to list format for PDF table
                headers = data.columns
                rows = []

                # Limit to first 20 rows for PDF
                for row_data in data.head(20).iter_rows():
                    rows.append(
                        [str(val) if val is not None else "" for val in row_data]
                    )

                if len(data) > 20:
                    pdf.set_font("Arial", "I", 9)
                    pdf.cell(0, 6, f"{title} (showing 20 of {len(data)} rows)", 0, 1)
                    pdf.ln(2)
                else:
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(0, 6, title, 0, 1)
                    pdf.ln(2)

                pdf.add_table(headers=headers, data=rows)

    # Add data preview
    pdf.add_section_title("Data Preview")
    pdf.set_font("Arial", "I", 9)
    pdf.cell(0, 6, f"First 15 rows of {state.df.shape[0]:,} total rows", 0, 1)
    pdf.ln(2)

    # Convert DataFrame preview to table
    preview_df = state.df.head(15)
    headers = preview_df.columns[:8]  # Limit columns for PDF width
    rows = []

    for row_data in preview_df.select(headers).iter_rows():
        rows.append([str(val)[:15] if val is not None else "" for val in row_data])

    pdf.add_table(headers=headers, data=rows)

    # Add visualizations as images
    if state.visualizations:
        for idx, viz_info in enumerate(state.visualizations, 1):
            pdf.add_page()
            title = viz_info.get("title", f"Visualization {idx}")
            fig = viz_info["figure"]

            pdf.add_section_title(title)

            # Convert plotly figure to image
            try:
                # Save figure as temporary image
                temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                fig.write_image(temp_img.name, width=1200, height=600)

                # Add to PDF
                pdf.image(temp_img.name, x=10, y=pdf.get_y(), w=190)

                # Clean up temp file
                os.unlink(temp_img.name)

            except Exception as e:
                pdf.set_font("Arial", "I", 10)
                pdf.cell(0, 8, f"[Visualization could not be rendered: {str(e)}]", 0, 1)

    # Add exports info
    if state.exports:
        pdf.add_page()
        pdf.add_section_title("Exported Files")
        pdf.set_font("Arial", "", 10)

        for export_path in state.exports:
            pdf.cell(0, 8, f"• {export_path}", 0, 1)

    # Save PDF
    pdf.output(str(filepath))

    return str(filepath)


# ============================================================
# Legacy function for backward compatibility
# ============================================================


def generate_pdf_report(
    analysis_text: str,
    image_paths: Optional[List[str]] = None,
    custom_filename: Optional[str] = None,
    company_name: str = "",
    tables: Optional[List[Dict[str, Any]]] = None,
    key_metrics: Optional[Dict[str, str]] = None,
) -> str:
    """
    Legacy function - generates a simple PDF report.

    For new code, use generate_report_from_state() instead.

    Args:
        analysis_text: The main analysis text
        image_paths: List of paths to images/charts to include
        custom_filename: Custom filename for the PDF (optional)
        company_name: Company name to display in header
        tables: List of table dicts with 'headers', 'data', and optional 'title'
        key_metrics: Dictionary of key metrics to display as cards

    Returns:
        str: Path to the generated PDF file
    """

    # Filename handling
    if custom_filename:
        filename = (
            custom_filename
            if custom_filename.endswith(".pdf")
            else f"{custom_filename}.pdf"
        )
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.pdf"

    # Create output directory
    output_path = Path("outputs")
    output_path.mkdir(exist_ok=True)
    filepath = output_path / filename

    # Create PDF
    pdf = CompanyReport(company_name=company_name)
    pdf.add_page()

    # Date
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(100, 100, 100)
    date_str = datetime.date.today().strftime("%d/%m/%Y")
    pdf.cell(0, 8, f"Generated on: {date_str}", 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Key metrics (if provided)
    if key_metrics:
        pdf.add_section_title("Key Metrics")
        pdf.add_key_metrics(key_metrics)

    # Main analysis
    pdf.add_section_title("Analysis and Comments")
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, analysis_text)
    pdf.ln(5)

    # Tables (if provided)
    if tables:
        for table in tables:
            if "title" in table:
                pdf.add_section_title(table["title"])

            pdf.add_table(
                headers=table["headers"],
                data=table["data"],
                col_widths=table.get("col_widths"),
            )

    # Images/Graphs
    if image_paths:
        for idx, img in enumerate(image_paths, 1):
            if os.path.exists(img):
                pdf.add_page()
                pdf.add_section_title(f"Graph {idx} - {os.path.basename(img)}")
                pdf.image(img, x=10, y=pdf.get_y(), w=190)
            else:
                print(f"Warning: Image not found: {img}")

    # Save PDF
    pdf.output(str(filepath))
    print(f"Report saved as: {filepath}")

    return str(filepath)


# ============================================================
# Example Usage
# ============================================================


if __name__ == "__main__":
    # Example usage of legacy function
    print("Testing legacy PDF report generation...")

    analysis_text = """
The Q4 2024 analysis shows strong performance across all business units.

Key Findings:
- Revenue growth exceeded projections by 12%
- Operating margin improved to 18.5%, up from 15.2% in Q3
- Customer acquisition costs decreased by 8%
- New product lines contributed 22% of total revenue

Recommendations:
- Increase investment in digital marketing channels
- Expand operations in emerging markets
- Strengthen partnerships with key distributors
- Launch customer loyalty program in Q1 2025
    """

    key_metrics = {
        "Revenue": "€2.5M",
        "Growth": "+15%",
        "Margin": "18.5%",
        "Customers": "245",
    }

    tables = [
        {
            "title": "Quarterly Performance",
            "headers": ["Quarter", "Revenue (€M)", "Margin (%)", "Customers"],
            "data": [
                ["Q1 2024", "1.8", "14.2", "1,250"],
                ["Q2 2024", "2.0", "15.1", "1,420"],
                ["Q3 2024", "2.2", "15.2", "1,680"],
                ["Q4 2024", "2.5", "18.5", "1,925"],
            ],
        }
    ]

    report_path = generate_pdf_report(
        analysis_text=analysis_text,
        custom_filename="test_report",
        company_name="Test Company",
        tables=tables,
        key_metrics=key_metrics,
    )

    print(f"✅ Test report generated: {report_path}")
