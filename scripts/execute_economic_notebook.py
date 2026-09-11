"""Execute, smoke-test dates, export notebook 09 without altering notebook 08."""
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import json
import hashlib
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from nbconvert import HTMLExporter

root=Path(__file__).resolve().parents[1]
path=root/'09_economic_vulnerability.ipynb'
nb=nbformat.read(path,as_version=4)
static='--static' in sys.argv
if static:
    nb.metadata.pop('widgets',None)
    for c in nb.cells:
        if c.cell_type=='code':
            c.outputs=[];c.execution_count=None
            c.source=c.source.replace('import ipywidgets as W',"raise ImportError('QA: optional widgets unavailable')")
nb.cells.append(nbformat.v4.new_code_cell('''# Temporary smoke check, removed after execution.
if W:
    original=E.predictions.copy(deep=True)
    for d in ['1990-03-31','2009-03-31','2020-03-31','2022-12-31']:
        cursor.value=d
        assert 'Conditional response' in P.assessment(d)
        assert 'Reconstructed IWV portfolio' in B.card(d)
    assert 'Unavailable' in B.card('1990-03-31')
    assert 'membership/weight updates' in B.card('2022-12-31')
    pd.testing.assert_frame_equal(E.predictions,original)
print('PASS: historical date controls and immutable evaluation')
'''))
with TemporaryDirectory(prefix='momentum-v9-') as folder:
    km=KernelManager(kernel_name='python3',transport='ipc',ip=str(Path(folder)/'kernel'))
    km.kernel_spec.argv[0]=sys.executable
    client=NotebookClient(nb,km=km,timeout=300,resources={'metadata':{'path':str(root)}},store_widget_state=True)
    try:client.execute()
    finally:
        if km.has_kernel:km.shutdown_kernel(now=True)
        if client.kc is not None:client.kc.stop_channels()
test=nb.cells.pop()
print(''.join(o.get('text','') for o in test.get('outputs',[]) if o.output_type=='stream'))
assert not [o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
if static:
    outputs=[o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[])]
    images=sum('image/png' in o.get('data',{}) for o in outputs)
    assert images==10,images
    result=dict(status='PASS',no_widget_figures=images,code_cells=sum(c.cell_type=='code' for c in nb.cells),
                notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                note='No-widget execution and separate live-kernel callback smoke checks; no browser-specific frontend certification')
    (root/'research/economic_extension/notebook_QA.json').write_text(json.dumps(result,indent=2))
    print('PASS: no-widget fallback; ten static figures; delivered notebook unchanged')
else:
    nbformat.write(nb,path)
    html,_=HTMLExporter().from_notebook_node(
        nb, resources={'metadata': {'name': 'US equity momentum: a statistical risk assessment'}})
    path.with_suffix('.html').write_text(html)
    print('PASS: notebook execution and HTML export')
