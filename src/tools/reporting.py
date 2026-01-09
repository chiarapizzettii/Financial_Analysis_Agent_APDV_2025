from fpdf import FPDF
import datetime
import os

class CompanyReport(FPDF):
    def __init__(self, titles):
        super().__init__()
        self.titles = titles

    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, self.titles['header'], 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f"{self.titles['page']} {self.page_no()}", 0, 0, 'C')

def generate_pdf_report(analysis_text, lang = "English", image_paths=None,  custom_filename=None):
    # 1. File name
    translations = {
        'Italiano': {'header': 'Report Analisi Aziendale', 'date': 'Data:', 'comments': 'Commenti e Analisi:', 'page': 'Pagina', 'graph': 'Grafico:'},
        'English': {'header': 'Business Analysis Report', 'date': 'Date:', 'comments': 'Comments and Analysis:', 'page': 'Page', 'graph': 'Graph:'},
        'Français': {'header': 'Rapport d\'Analyse', 'date': 'Date:', 'comments': 'Commentaires:', 'page': 'Page', 'graph': 'Graphique:'},
        'Español': {'header': 'Informe de Análisis', 'date': 'Fecha:', 'comments': 'Comentarios:', 'page': 'Página', 'graph': 'Gráfico:'}
    }

    t = translations.get(lang, translations['English'])

    if custom_filename:
        filename = custom_filename if custom_filename.lower().endswith('.pdf') else f"{custom_filename}.pdf"
    else:
        filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

    pdf = CompanyReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Date
    pdf.cell(0, 10, f"Datee: {datetime.date.today()}", 0, 1)
    pdf.ln(10)
    
    # Insert comments
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Comments and Analysis:", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, analysis_text)
    
    # Graphs 
    if image_paths:
        for img in image_paths:
            if os.path.exists(img):
                pdf.add_page() 
                pdf.cell(0, 10, f"{t['graph']} {os.path.basename(img)}", 0, 1)
                pdf.image(img, x=10, y=30, w=190)
            
    pdf.output(filename)
    return filename




# #### IN STREAMLIT APP ####
# FOR BOTH REPORTING AND LANGUAGE DETECTOR

# import streamlit as st
# from tools.language_detector import get_user_language
# from tools.report_tool import generate_pdf_report

# # ... inside your main chat loop ...
# if prompt := st.chat_input("Ask about company data..."):
#     # 1. Detect language
#     detected_lang = get_user_language(prompt)
#     st.session_state.lang = detected_lang # Store it for the PDF later

#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         # 2. Instruct the agent to reply in the detected language
#         # Pass the language into your agent's reasoning process
#         full_query = f"The user is speaking {detected_lang}. Reply in {detected_lang}: {prompt}"
#         response = agent.run(full_query) 
        
#         st.markdown(response)
#         st.session_state.last_response = response # Save for the report

# # 3. Report Generation Button
# if st.button("Download PDF Report"):
#     if "last_response" in st.session_state:
#         # Pass the saved language and response to the tool
#         report_file = generate_pdf_report(
#             analysis_text=st.session_state.last_response,
#             lang=st.session_state.get("lang", "English"),
#             custom_filename="Company_Analysis"
#         )
        
#         with open(report_file, "rb") as f:
#             st.download_button("Click here to download", f, file_name=report_file)