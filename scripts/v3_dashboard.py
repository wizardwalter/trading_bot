#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'v3_btc_accumulator' / 'out'
LATEST = OUT / 'latest_v3.json'
RUNS = OUT / 'training_runs.jsonl'

HTML = '''<!doctype html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/><title>V3 BTC Accumulator</title>
<style>body{font-family:Inter,system-ui;background:#0b1020;color:#e6ecff;margin:0}.w{max-width:1250px;margin:18px auto;padding:0 14px}.g{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.c{background:#121a33;border:1px solid #263159;border-radius:14px;padding:12px}.k{font-size:12px;color:#95a4d8}.v{font-size:24px;font-weight:700}.row{display:grid;grid-template-columns:2fr 1fr;gap:12px;margin-top:12px}canvas{width:100%;height:280px;background:#121a33;border:1px solid #263159;border-radius:14px}.ok{color:#53d18c}.bad{color:#ff7272}</style></head>
<body><div class="w"><h1>₿ V3 BTC Accumulator</h1><div class="g">
<div class="c"><div class="k">Latest End BTC</div><div id="end" class="v">-</div></div>
<div class="c"><div class="k">Latest ΔBTC</div><div id="delta" class="v">-</div></div>
<div class="c"><div class="k">Accumulation %</div><div id="pct" class="v">-</div></div>
<div class="c"><div class="k">Core Floor</div><div id="core" class="v">-</div></div></div>
<div class="row"><canvas id="btcChart"></canvas><div class="c"><div class="k">Latest Summary</div><div id="sum" style="line-height:1.7;margin-top:8px"></div><hr style="border-color:#2a355f"><div class="k">Best/Worst Run</div><div id="bw" style="line-height:1.7;margin-top:8px"></div></div></div>
<canvas id="deltaChart" style="margin-top:12px"></canvas></div>
<script>
const f=(n,d=6)=> (n===null||n===undefined||isNaN(n))?'n/a':Number(n).toFixed(d);
const fp=(n)=> (n===null||n===undefined||isNaN(n))?'n/a':Number(n).toFixed(2)+'%';
function draw(id,vals,color,label){const c=document.getElementById(id),ctx=c.getContext('2d');const W=c.width=c.clientWidth*devicePixelRatio,H=c.height=c.clientHeight*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);ctx.clearRect(0,0,c.clientWidth,c.clientHeight);if(!vals.length)return;const max=Math.max(...vals),min=Math.min(...vals);const lo=min===max?min-1e-6:min,hi=min===max?max+1e-6:max;const y=v=>c.clientHeight-((v-lo)/(hi-lo))*c.clientHeight;ctx.strokeStyle='#2a355f';ctx.beginPath();ctx.moveTo(0,y(0));ctx.lineTo(c.clientWidth,y(0));ctx.stroke();ctx.beginPath();vals.forEach((v,i)=>{const x=i*(c.clientWidth/(vals.length-1||1)); if(i===0)ctx.moveTo(x,y(v)); else ctx.lineTo(x,y(v));});ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='#9cb0e6';ctx.font='12px Inter';ctx.fillText(label,8,14)}
async function pull(){const l=await fetch('/api/latest').then(r=>r.json());const rs=await fetch('/api/runs').then(r=>r.json());const rows=rs.runs||[];document.getElementById('end').textContent=f(l.end_btc,6);document.getElementById('delta').textContent=(l.delta_btc>=0?'+':'')+f(l.delta_btc,6);document.getElementById('delta').className='v '+((l.delta_btc||0)>=0?'ok':'bad');document.getElementById('pct').textContent=fp(l.btc_accum_pct);document.getElementById('pct').className='v '+((l.btc_accum_pct||0)>=0?'ok':'bad');document.getElementById('core').textContent=fp((l.core_btc_floor||0)*100);
const roll = rows.slice(-20); const avg = roll.length?roll.reduce((a,b)=>a+(b.btc_accum_pct||0),0)/roll.length:0;const best=[...rows].sort((a,b)=>(b.delta_btc||0)-(a.delta_btc||0))[0]||{};const worst=[...rows].sort((a,b)=>(a.delta_btc||0)-(b.delta_btc||0))[0]||{};
document.getElementById('sum').innerHTML=`run_id: <b>${l.run_id||'n/a'}</b><br>trades: <b>${l.trades||0}</b> (trims ${l.trims||0} / buys ${l.buys||0})<br>win_rate: <b>${fp((l.win_rate||0)*100)}</b><br>max BTC DD: <b>${fp((l.max_btc_drawdown||0)*100)}</b><br>fees BTC: <b>${f(l.fees_paid_btc,6)}</b><br>slippage BTC: <b>${f(l.slippage_est_btc,6)}</b><br>rolling avg accum (20): <b>${fp(avg)}</b>`;
document.getElementById('bw').innerHTML=`best ΔBTC: <b>${f(best.delta_btc,6)}</b><br>worst ΔBTC: <b>${f(worst.delta_btc,6)}</b>`;
draw('btcChart', rows.map(x=>x.end_btc||0), '#71a7ff', 'Total BTC per run');
draw('deltaChart', rows.map(x=>x.delta_btc||0), '#41d37b', 'Delta BTC per run');}
pull();setInterval(pull,10000);
</script></body></html>'''

class H(BaseHTTPRequestHandler):
    def _j(self, p: Path):
        if not p.exists(): return {}
        try: return json.loads(p.read_text())
        except Exception: return {}
    def do_GET(self):
        p = urlparse(self.path).path
        if p in ['/', '/index.html']:
            self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.end_headers(); self.wfile.write(HTML.encode()); return
        if p == '/api/latest':
            self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(json.dumps(self._j(LATEST)).encode()); return
        if p == '/api/runs':
            runs=[]
            if RUNS.exists():
                for line in RUNS.read_text().splitlines()[-300:]:
                    try:runs.append(json.loads(line))
                    except:pass
            self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(json.dumps({'runs':runs}).encode()); return
        self.send_response(404); self.end_headers()

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--host',default='127.0.0.1'); ap.add_argument('--port',type=int,default=8510); a=ap.parse_args()
    s=ThreadingHTTPServer((a.host,a.port),H); print(f'V3 dashboard: http://{a.host}:{a.port}'); s.serve_forever()
