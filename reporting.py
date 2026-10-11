"""Shared local reporting utilities; no external assets or tracking."""
import hashlib, html, json, platform, sys, time, urllib.request
from pathlib import Path

STYLE = """body{font:16px/1.55 Arial,Helvetica,sans-serif;color:#171717;background:white;max-width:1120px;margin:42px auto;padding:0 24px}h1{font-size:38px;letter-spacing:-1px;margin-bottom:8px}h2{font-size:23px;margin-top:40px;border-bottom:1px solid #d8d8d4;padding-bottom:10px}p,small{color:#404040}a{color:#1f6f5f}.eyebrow{font-size:13px;text-transform:uppercase;letter-spacing:2px;color:#1f6f5f}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}th,td{padding:10px 12px;text-align:right;border-bottom:1px solid #e4e7e5;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{font-weight:600;border-bottom:1px solid #888}input,select,button{font:inherit;padding:9px;border:1px solid #aaa;border-radius:3px}input{width:min(90%,440px)}button{background:#1f6f5f;color:white;cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f2;padding:18px}footer{margin-top:45px;border-top:1px solid #d8d8d4;padding-top:18px;font-size:13px}svg{max-width:100%}.numbers{display:flex;gap:45px;flex-wrap:wrap;border-block:1px solid #d8d8d4;padding:20px 0}.numbers b{font-size:28px;display:block}.numbers span{font-size:14px;color:#404040}@media(max-width:650px){body{margin:24px auto;padding:0 16px}h1{font-size:30px}.numbers{gap:20px}th,td{padding:7px}}"""

def download(url,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        temp=path.with_suffix(path.suffix+'.part')
        try:
            with urllib.request.urlopen(url,timeout=180) as r, temp.open('wb') as f:
                while chunk:=r.read(1024*1024): f.write(chunk)
            temp.replace(path)
        finally: temp.unlink(missing_ok=True)
    return path

def environment():
    cpu=platform.processor() or platform.machine()
    if Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'): cpu=line.split(':',1)[1].strip();break
    return {'python':sys.version.split()[0],'platform':platform.platform(),'cpu':cpu}

def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

LABELS = {'model': 'Model', 'ndcg10': 'nDCG@10 ↑', 'recall10': 'Recall@10 ↑', 'mrr10': 'MRR@10 ↑', 'ndcg10_ci_low': '95% CI lower', 'ndcg10_ci_high': '95% CI upper', 'query_p50_ms': 'Query P50 (ms) ↓', 'query_p95_ms': 'Query P95 (ms) ↓', 'test_accuracy': 'Test accuracy ↑', 'accuracy': 'Accuracy ↑', 'macro_f1': 'Macro-F1 ↑', 'accuracy_ci_low': '95% CI lower', 'accuracy_ci_high': '95% CI upper', 'noise_accuracy': 'Noise accuracy ↑', 'checkpoint_bytes': 'Checkpoint (bytes) ↓', 'p50_ms': 'P50 (ms) ↓', 'p95_ms': 'P95 (ms) ↓', 'batch256_images_per_second': 'Batch-256 images/s ↑', 'validation_macro_f1': 'Validation macro-F1 ↑', 'fit_seconds': 'Fit time (s)', 'log_loss': 'Log loss ↓', 'coverage': 'Coverage', 'accepted_accuracy': 'Accepted accuracy ↑', 'failed_records': 'Failed records', 'status': 'Status', 'severity': 'Severity', 'rule': 'Rule', 'description': 'Definition', 'safe_payments_brl': 'Payment-grain total (BRL)', 'naive_join_payments_brl': 'Naive join total (BRL)', 'month': 'Month', 'orders': 'Orders', 'delivery_denominator': 'Eligible deliveries', 'late_pct': 'Late deliveries (%)', 'merchandise_brl': 'Merchandise (BRL)', 'seller_id': 'Seller', 'seller_state': 'Seller state', 'customer_state': 'Destination state', 'late_orders': 'Late orders', 'review_score': 'Mean review score', 'dispatch_days': 'Dispatch (days)', 'transit_days': 'Transit (days)', 'is_late': 'Late delivery?', 'orders_with_review': 'Orders with review', 'mean_review_score': 'Mean review score', 'training_loss': 'Training loss', 'validation_accuracy': 'Validation accuracy ↑', 'weight': 'Dense weight', 'rrf_k': 'RRF constant', 'train_ndcg10': 'Train nDCG@10 ↑', 'feature': 'Feature', 'validation_f1_drop': 'Validation F1 decrease', 'class': 'Class', 'precision': 'Precision', 'recall': 'Recall', 'f1-score': 'F1', 'support': 'Test samples', 'epoch': 'Epoch'}

def table(rows,columns):
    def cell(x):
        if x is None:return '—'
        if isinstance(x,float):return f'{x:,.4f}'
        if isinstance(x,int):return f'{x:,}'
        return html.escape(str(x))
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(LABELS.get(c,c.replace('_',' ').capitalize()))+'</th>' for c in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+cell(r.get(c))+'</td>' for c in columns)+'</tr>' for r in rows)+'</tbody></table></div>'

def report(path,title,subtitle,sections,metrics=None):
    nums=''
    if metrics:
        nums='<div class="numbers">'+''.join('<div><b>'+html.escape(str(v))+'</b><span>'+html.escape(k)+'</span></div>' for k,v in metrics.items())+'</div>'
    document='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+STYLE+'</style><body><p class="eyebrow">Reproducible portfolio / evidence report</p><h1>'+html.escape(title)+'</h1><p>'+html.escape(subtitle)+'</p>'+nums+'<p><label>Filter table rows <input id="filter" placeholder="Search names, dates or metrics"></label></p>'+''.join('<section><h2>'+html.escape(t)+'</h2>'+v+'</section>' for t,v in sections)+'<footer>Generated from the accompanying pipeline. See README and results/metrics.json for definitions, sample sizes and limitations. No externally hosted scripts.</footer><script>document.querySelector("#filter").addEventListener("input",e=>{let q=e.target.value.toLowerCase();document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))})</script></body></html>'
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(document)
