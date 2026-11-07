"""Export every saved notebook cell and output to a readable, page-counted PDF."""
from pathlib import Path
from io import BytesIO
from html.parser import HTMLParser
import json,html,re,textwrap,base64,hashlib
from reportlab.platypus import SimpleDocTemplate,Paragraph,Preformatted,Spacer,Image,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent

class DataTable(HTMLParser):
    def __init__(self):super().__init__();self.rows=[];self.row=[];self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='tr':self.row=[]
        if tag in ('td','th'):self.cell=[]
    def handle_data(self,data):
        if self.cell is not None:self.cell.append(data)
    def handle_endtag(self,tag):
        if tag in ('td','th') and self.cell is not None:
            self.row.append(' '.join(''.join(self.cell).split()));self.cell=None
        if tag=='tr' and self.row:self.rows.append(self.row)

def inline(s):
    s=html.escape(s)
    s=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'\1 (\2)',s)
    return s.replace('$','')

def export_notebook():
    path=ROOT/'Yash_Desai_HW6.ipynb';nb=json.loads(path.read_text())
    body=ParagraphStyle('body',fontName='Helvetica',fontSize=10,leading=13,spaceAfter=6)
    heads={1:ParagraphStyle('h1',fontName='Helvetica-Bold',fontSize=16,leading=19,spaceAfter=8,keepWithNext=True),
           2:ParagraphStyle('h2',fontName='Helvetica-Bold',fontSize=12,leading=15,spaceBefore=9,spaceAfter=5,keepWithNext=True),
           3:ParagraphStyle('h3',fontName='Helvetica-Bold',fontSize=10,leading=13,spaceBefore=6,spaceAfter=4,keepWithNext=True)}
    code=ParagraphStyle('code',fontName='Courier',fontSize=8.5,leading=10.4,spaceAfter=5)
    cell=ParagraphStyle('cell',fontName='Helvetica',fontSize=7.8,leading=10)
    label=ParagraphStyle('label',fontName='Helvetica',fontSize=8,leading=10,textColor=colors.HexColor('#536a7b'),keepWithNext=True)
    story=[];images=0;input_lines=0;output_blocks=0
    for c in nb['cells']:
        source=''.join(c['source'])
        if c['cell_type']=='markdown':
            for block in source.split('\n\n'):
                if not block.strip():continue
                lines=block.splitlines();first=lines[0]
                if first.startswith('#'):
                    n=len(first)-len(first.lstrip('#'))
                    story.append(Paragraph(inline(first[n:].strip()),heads[min(n,3)]))
                    if len(lines)>1:story.append(Paragraph(inline(' '.join(lines[1:])),body))
                else:story.append(Paragraph(inline(' '.join(lines)),body))
        else:
            input_lines+=len(source.splitlines())
            story.append(Paragraph('Code cell '+str(c.get('execution_count','')),label))
            wrapped='\n'.join('\n'.join(textwrap.wrap(line,width=103,replace_whitespace=False,drop_whitespace=False,subsequent_indent='    ')) if line else '' for line in source.splitlines())
            story.append(Preformatted(wrapped,code))
            for o in c.get('outputs',[]):
                data=o.get('data',{});output_blocks+=1
                if 'image/png' in data:
                    raw=data['image/png'];raw=''.join(raw) if isinstance(raw,list) else raw
                    stream=BytesIO(base64.b64decode(raw));w,h=ImageReader(stream).getSize()
                    ratio=min(480/w,360/h)
                    story.append(Image(stream,width=w*ratio,height=h*ratio));images+=1
                elif 'text/html' in data:
                    raw=data['text/html'];raw=''.join(raw) if isinstance(raw,list) else raw
                    parser=DataTable();parser.feed(raw)
                    if parser.rows:
                        cols=max(map(len,parser.rows));widths=[72]+[(540-72)/(cols-1)]*(cols-1) if cols>1 else [540]
                        rows=[[Paragraph(html.escape(v),cell) for v in row+['']*(cols-len(row))] for row in parser.rows]
                        table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
                        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf2f6')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3),('LINEBELOW',(0,0),(-1,0),.4,colors.grey)]))
                        story.append(table)
                    else:story.append(Preformatted(html.unescape(re.sub('<[^>]+>','',raw)),code))
                else:
                    raw=o.get('text',data.get('text/plain',''));raw=''.join(raw) if isinstance(raw,list) else raw
                    if raw:
                        wrapped='\n'.join('\n'.join(textwrap.wrap(l,width=103,replace_whitespace=False,drop_whitespace=False)) if l else '' for l in raw.splitlines())
                        story.append(Preformatted(wrapped,code))
                story.append(Spacer(1,4))
    dest=ROOT/'results/HW6_Notebook_Print.pdf'
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.drawRightString(576,20,str(doc.page))
    SimpleDocTemplate(str(dest),pagesize=(612,792),leftMargin=36,rightMargin=36,topMargin=32,bottomMargin=32,
                      title='Homework 6 - complete compact notebook export').build(story,onFirstPage=footer,onLaterPages=footer)
    pages=len(PdfReader(str(dest)).pages)
    check={'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pages':pages,'page_size':'US Letter',
           'body_font_pt':10,'code_font_pt':8.5,'notebook_cells':len(nb['cells']),
           'input_lines_included':input_lines,'embedded_figures_included':images,'output_blocks_included':output_blocks,
           'code_cells_hidden':False,'live_browser_preview_verified':False,
           'scope':'This PDF renders every cell and saved output of the compact notebook; arbitrary browser/GitHub styles can paginate differently.'}
    (ROOT/'results/notebook_export_check.json').write_text(json.dumps(check,indent=2))
    print(json.dumps(check,indent=2))
    if pages>20:raise RuntimeError(f'Notebook export has {pages} pages; reduce repetition or revise layout.')
    return dest

if __name__=='__main__':export_notebook()
