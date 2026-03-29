#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DIR = ROOT / "data" / "backtests"

HTML = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>BTC Training Mission Control</title>
  <style>
    body { font-family: Inter, system-ui, sans-serif; margin:0; background:#0b1020; color:#e6ecff; }
    .wrap { max-width: 1200px; margin: 24px auto; padding: 0 16px; }
    h1 { margin: 0 0 12px; }
    .grid { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; }
    .card { background:#121a33; border:1px solid #263159; border-radius:14px; padding:14px; }
    .k { font-size:12px; color:#95a4d8; }
    .v { font-size:24px; font-weight:700; }
    canvas { width:100%; height:280px; background:#121a33; border:1px solid #263159; border-radius:14px; margin-top:12px; }
    .row { display:grid; grid-template-columns: 2fr 1fr; gap:12px; margin-top:12px; }
    .ok { color:#53d18c; } .bad { color:#ff7272; } .warn{ color:#ffcb6b; }
    @media (max-width: 900px) { .grid { grid-template-columns: repeat(2,1fr)} .row{grid-template-columns:1fr;} }
  </style>
</head>
<body>
<div class=\"wrap\">
  <h1>🧠 BTC Training Mission Control</h1>
  <div class=\"grid\">
    <div class=\"card\"><div class=\"k\">Profile / Variant</div><div id=\"profile\" class=\"v\">-</div></div>
    <div class=\"card\"><div class=\"k\">Latest Test Return</div><div id=\"ret\" class=\"v\">-</div></div>
    <div class=\"card\"><div class=\"k\">Profit Factor</div><div id=\"pf\" class=\"v\">-</div></div>
    <div class=\"card\"><div class=\"k\">Do Not Trade</div><div id=\"dnt\" class=\"v\">-</div></div>
  </div>

  <div class=\"row\">
    <canvas id=\"retChart\"></canvas>
    <div class=\"card\">
      <div class=\"k\">Rolling Window Snapshot</div>
      <div id=\"roll\" style=\"margin-top:8px; line-height:1.7\">-</div>
      <hr style=\"border-color:#2a355f\" />
      <div class=\"k\">Training Pulse</div>
      <div id=\"pulse\" class=\"v\" style=\"font-size:18px\">waiting...</div>
    </div>
  </div>
</div>
<script>
const fmtPct = (v)=> (v===null||v===undefined||isNaN(v)) ? 'n/a' : (v*100).toFixed(2)+'%';
const fmt = (v)=> (v===null||v===undefined||isNaN(v)) ? 'n/a' : Number(v).toFixed(2);
let lastTs = null;

async function pull(){
  const [latestRes, shadowRes] = await Promise.all([
    fetch('/api/latest').then(r=>r.json()),
    fetch('/api/shadow').then(r=>r.json())
  ]);

  const test = latestRes.test || {};
  document.getElementById('profile').textContent = `${latestRes.training_profile||'-'} / ${latestRes.variant||'-'}`;
  document.getElementById('ret').textContent = fmtPct(test.total_return);
  document.getElementById('ret').className = 'v ' + ((test.total_return||0)>=0 ? 'ok':'bad');
  document.getElementById('pf').textContent = fmt(test.profit_factor);
  document.getElementById('dnt').textContent = String(test.do_not_trade);
  document.getElementById('dnt').className = 'v ' + (test.do_not_trade ? 'warn':'ok');

  const hist = (shadowRes.history||[]).filter(x=>x.profile==='neural').slice(-120);
  drawChart(hist.map(h=>h.ret||0));

  const win = hist.slice(-24);
  const avg = arr=> arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0;
  const pos = win.filter(x=>(x.ret||0)>0).length;
  const avgRet = avg(win.map(x=>x.ret||0));
  const avgDd = avg(win.map(x=>x.dd||0));
  const avgTrades = avg(win.map(x=>x.trades||0));
  document.getElementById('roll').innerHTML =
    `24-run avg ret: <b>${fmtPct(avgRet)}</b><br>`+
    `24-run pos ratio: <b>${win.length?Math.round(100*pos/win.length):0}%</b><br>`+
    `24-run avg drawdown: <b>${fmtPct(avgDd)}</b><br>`+
    `24-run avg trades: <b>${avgTrades.toFixed(1)}</b>`;

  const latestTs = (shadowRes.latest||{}).ts || null;
  const pulse = document.getElementById('pulse');
  if (latestTs && latestTs !== lastTs) {
    pulse.textContent = 'new iteration detected ✅';
    pulse.className = 'v ok';
    lastTs = latestTs;
  } else {
    pulse.textContent = 'watching for next run...';
    pulse.className = 'v';
  }
}

function drawChart(vals){
  const c=document.getElementById('retChart');
  const ctx=c.getContext('2d');
  const w=c.width=c.clientWidth*devicePixelRatio;
  const h=c.height=c.clientHeight*devicePixelRatio;
  ctx.scale(devicePixelRatio,devicePixelRatio);
  ctx.clearRect(0,0,c.clientWidth,c.clientHeight);
  const W=c.clientWidth,H=c.clientHeight;
  ctx.strokeStyle='#2a355f'; ctx.beginPath(); ctx.moveTo(0,H/2); ctx.lineTo(W,H/2); ctx.stroke();
  if(!vals.length) return;
  const max=Math.max(...vals,0.001), min=Math.min(...vals,-0.001);
  const y=v=> H-((v-min)/(max-min))*H;
  ctx.beginPath();
  vals.forEach((v,i)=>{
    const x=i*(W/(vals.length-1||1));
    if(i===0) ctx.moveTo(x,y(v)); else ctx.lineTo(x,y(v));
  });
  ctx.strokeStyle='#71a7ff'; ctx.lineWidth=2; ctx.stroke();
}

pull();
setInterval(pull, 15000);
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, path: Path) -> bytes:
        if not path.exists():
            return b"{}"
        try:
            return path.read_bytes()
        except Exception:
            return b"{}"

    def do_GET(self):
        p = urlparse(self.path).path
        if p in {"/", "/index.html"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
            return

        if p == "/api/latest":
            payload = self._json(BACKTEST_DIR / "latest.json")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)
            return

        if p == "/api/shadow":
            payload = self._json(BACKTEST_DIR / "shadow_score.json")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(404)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Training dashboard web server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
