import datetime
import os

from fpdf import FPDF


class CompanyReport(FPDF):
    def __init__(self, titles):
        super().__init__()
        self.titles = titles

    def header(self):
        self.set_font("Arial", "B", 15)
        self.cell(0, 10, self.titles["header"], 0, 1, "C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"{self.titles['page']} {self.page_no()}", 0, 0, "C")


def generate_pdf_report(
    analysis_text, lang="English", image_paths=None, custom_filename=None
):
    """
    Generate a PDF with analysis and graphs.

    Args:
        analysis_text (str): The analysis text to be included.
        lang (str): The report language ('Italiano', 'English', 'Français', 'Español').
        image_paths (list): A list of paths to the images/charts to be included.
        custom_filename (str): A custom filename for the file (optional).

    Returns:
        str: File name of the generated PDF

    """

    translations = {
        "Italiano": {
            "header": "Report Analisi Aziendale",
            "date": "Data:",
            "comments": "Commenti e Analisi:",
            "page": "Pagina",
            "graph": "Grafico:",
        },
        "English": {
            "header": "Business Analysis Report",
            "date": "Date:",
            "comments": "Comments and Analysis:",
            "page": "Page",
            "graph": "Graph:",
        },
        "Français": {
            "header": "Rapport d'Analyse",
            "date": "Date:",
            "comments": "Commentaires et Analyse:",
            "page": "Page",
            "graph": "Graphique:",
        },
        "Español": {
            "header": "Informe de Análisis",
            "date": "Fecha:",
            "comments": "Comentarios y Análisis:",
            "page": "Página",
            "graph": "Gráfico:",
        },
    }

    t = translations.get(lang, translations["English"])

    if custom_filename:
        filename = (
            custom_filename
            if custom_filename.endswith(".pdf")
            else f"{custom_filename}.pdf"
        )
    else:
        filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    pdf = CompanyReport(titles=t)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"{t['date']} {datetime.date.today().strftime('%d/%m/%Y')}", 0, 1)
    pdf.ln(5)

    # Date
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, t["comments"], 0, 1)

    # Insert comments
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, analysis_text)
    pdf.ln(10)

    # Graphs
    if image_paths:
        for idx, img in enumerate(image_paths, 1):
            if os.path.exists(img):
                pdf.add_page()
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"{t['graph']} {idx} - {os.path.basename(img)}", 0, 1)
                pdf.ln(5)

                pdf.image(img, x=10, y=pdf.get_y(), w=190)
            else:
                print(f"Avviso: Image not found: {img}")

    # Save PDF
    pdf.output(filename)
    print(f"Report salvato come: {filename}")
    return filename


# ESEMPIO DI UTILIZZO
if __name__ == "__main__":
    # Esempio 1: Report in italiano con grafici
    testo_analisi = """
    L'analisi dei dati aziendali mostra un trend positivo nel Q4 2024.

    Punti chiave:
    - Le vendite sono aumentate del 15% rispetto al trimestre precedente
    - Il margine operativo è migliorato del 3%
    - Nuovi clienti acquisiti: 245

    Raccomandazioni:
    - Investire maggiormente nel marketing digitale
    - Espandere la presenza nei mercati emergenti
    """

    grafici = ["grafico_vendite.png", "grafico_margini.png"]

    # Genera report in italiano
    generate_pdf_report(
        analysis_text=testo_analisi,
        lang="Italiano",
        image_paths=grafici,
        custom_filename="report_Q4_2024",
    )

    # Esempio 2: Report in inglese senza grafici
    english_text = """
    The quarterly analysis reveals strong performance across all metrics.
    Revenue increased by 20% year-over-year.
    """

    generate_pdf_report(analysis_text=english_text, lang="English")
