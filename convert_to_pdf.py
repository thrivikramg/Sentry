import os
import markdown2
from xhtml2pdf import pisa

def convert_md_to_pdf(md_path, pdf_path):
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        return

    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert Markdown to HTML
    html_content = markdown2.markdown(md_content, extras=["tables", "fenced-code-blocks"])

    # Sanitize content for PDF compatibility (replace problematic unicode/math symbols)
    # Simple replacements to avoid font errors while keeping readability
    replacements = {
        '$\mathcal{T}$': 'T',
        '$$': '',
        '\mathcal{K}': 'K',
        '\text{Disparity}_{\text{bbox}}': 'Disparity_bbox',
        '\delta': 'delta',
        '\epsilon': 'epsilon',
        'c_x': 'cx',
        'p_x': 'px',
        'f_{pseudo}': 'f_pseudo',
        '\mathcal{P}': 'P',
        'DSS_{t}': 'DSS(t)',
        'DSS_{t-1}': 'DSS(t-1)',
        '\Delta_{\text{recovery}}': 'Delta_recovery',
        '\\alpha': 'alpha',
        '\\sigma': 'sigma',
        '\\vec{v}': 'vector_v',
        '\\vec{Dist}_{ego}': 'dist_ego',
        '\Delta t': 'delta_t',
        '\\text{TTC}': 'TTC',
        '\\mathbb{R}': 'R',
        '\in': 'in',
        '\\times': 'x',
    }
    for old, new in replacements.items():
        html_content = html_content.replace(old, new)

    # Wrap in basic HTML structure
    full_html = f"""
    <html>
    <head>
    <style>
        body {{ font-family: Helvetica, Arial, sans-serif; font-size: 10pt; }}
        h1 {{ font-size: 18pt; color: #2c3e50; text-align: center; }}
        h2 {{ font-size: 14pt; color: #2c3e50; border-bottom: 1px solid #eee; margin-top: 20px; }}
        h3 {{ font-size: 12pt; color: #34495e; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        pre {{ background-color: #f8f8f8; padding: 10px; border: 1px solid #ddd; font-family: Courier; }}
    </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """

    # Create PDF
    with open(pdf_path, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(full_html, dest=result_file)

    if not pisa_status.err:
        print(f"✅ Success: PDF generated at {pdf_path}")
    else:
        print(f"❌ Error during PDF generation")

if __name__ == "__main__":
    src = r"C:\Users\Thrivikram\.gemini\antigravity\brain\8a5cd577-c10a-401b-83c4-65be6f59a8b9\artifacts\comprehensive_phd_dissertation_report.md"
    dest = r"C:\Users\Thrivikram\.gemini\antigravity\brain\8a5cd577-c10a-401b-83c4-65be6f59a8b9\artifacts\comprehensive_phd_dissertation_report.pdf"
    convert_md_to_pdf(src, dest)
