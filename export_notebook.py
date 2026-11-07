"""Make a fixed-page browser-print copy of the verified homework PDF."""
from pathlib import Path
import base64,subprocess,tempfile
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent

def export():
    pdf=ROOT/'results/HW6_Notebook_Print.pdf'
    import hashlib,json
    manifest=ROOT/'results/notebook_export_check.json'
    digest=hashlib.sha256((ROOT/'Yash_Desai_HW6.ipynb').read_bytes()).hexdigest()
    if not pdf.exists() or not manifest.exists() or json.loads(manifest.read_text()).get('notebook_sha256')!=digest:
        from notebook_print import export_notebook
        export_notebook()
    count=len(PdfReader(str(pdf)).pages)
    if count>20:raise ValueError(f'The report has {count} pages; the limit is 20.')
    with tempfile.TemporaryDirectory(prefix='hw6_print_') as tmp:
        prefix=Path(tmp)/'page'
        subprocess.run(['pdftoppm','-r','144','-png',str(pdf),str(prefix)],check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        pages=sorted(Path(tmp).glob('page-*.png'))
        assert len(pages)==count
        images=[]
        for i,p in enumerate(pages,1):
            data=base64.b64encode(p.read_bytes()).decode()
            images.append(f'<section class="page" aria-label="Page {i} of {count}"><img alt="Homework PDF page {i} of {count}" src="data:image/png;base64,{data}"></section>')
    css='''@page{size:letter;margin:0}*{box-sizing:border-box}html,body{margin:0;padding:0}.page{display:block;width:8.5in;height:11in;overflow:hidden;background:white;break-after:page;page-break-after:always}.page:last-child{break-after:auto;page-break-after:auto}.page img{display:block;width:100%;height:100%}@media screen{body{background:#dfe5eb;font:15px Arial,sans-serif}.screen-only{max-width:8.5in;margin:20px auto;padding:15px;background:white}.page{margin:16px auto;box-shadow:0 2px 8px #999}}@media print{.screen-only{display:none}body{-webkit-print-color-adjust:exact;print-color-adjust:exact}}'''
    note=f'''<div class="screen-only"><strong>Homework 6: {count} fixed pages</strong><p>This browser copy mirrors the complete compact notebook PDF, including every code cell, answer and saved output. It is an image-based copy; use the PDF for selectable text and the notebook for interactive code.</p><p>Print on US Letter at 100% scale, with browser headers and footers off. Live browser pagination could not be tested in the preparation environment. Each source PDF page is placed in its own fixed-size print section. Printing the native notebook or GitHub page can produce a different page count.</p></div>'''
    dest=ROOT/'Yash_Desai_HW6_print.html'
    dest.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Homework 6 - fixed-page print copy</title><style>'+css+'</style></head><body>'+note+''.join(images)+'</body></html>')
    return dest,count

if __name__=='__main__':
    p,n=export();print(p);print('Fixed print sections:',n)
