import PyPDF2
from typing import List

def extract_text_from_pdf(pdf_file: str) -> List[str]:
    with open(pdf_file, "rb") as pdf:
        reader = PyPDF2.PdfReader(pdf, strict=False)
        pdf_text = []
        for page in reader.pages:
            content = page.extract_text()
            pdf_text.append(content)

        return pdf_text
    
if __name__ == '__main__':
    extracted_text = extract_text_from_pdf('data/resume.pdf')
    with open('resume.txt', 'w', encoding='utf-8') as f:
        for text in extracted_text:
            f.write(text)