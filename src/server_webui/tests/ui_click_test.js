// Drive the served page in headless Chrome: custom size + LoRA checkboxes, then one real generation.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');
const WebSocket = require('/usr/share/nodejs/ws');
const [base, profile, shot] = process.argv.slice(2);
const delay = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  const chrome = spawn('google-chrome', ['--headless=new', '--no-sandbox', '--disable-gpu', '--no-first-run',
    '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--window-size=1000,2400', base], { stdio: 'ignore' });
  try {
    const portFile = path.join(profile, 'DevToolsActivePort');
    while (!fs.existsSync(portFile)) await delay(100);
    const [port] = fs.readFileSync(portFile, 'utf8').trim().split('\n');
    let page;
    for (let i = 0; !page && i < 50; i++) {
      page = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find(t => t.type === 'page' && t.url.startsWith(base));
      if (!page) await delay(100);
    }
    const ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((res, rej) => { ws.once('open', res); ws.once('error', rej); });
    let id = 0; const pending = new Map();
    ws.on('message', raw => { const m = JSON.parse(raw); if (pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
    const send = (method, params = {}) => new Promise(res => { pending.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
    const run = async expr => {
      const m = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
      if (m.result.exceptionDetails) throw new Error(JSON.stringify(m.result.exceptionDetails));
      return m.result.result.value;
    };
    const waitFor = async (expr, ms) => { for (const end = Date.now() + ms; Date.now() < end; await delay(250)) if (await run(expr)) return; throw new Error('timeout: ' + expr); };

    await waitFor(`document.getElementById('controls')?.disabled === false`, 15000);
    await run(`(() => {
      window.sent = [];
      const real = window.fetch;
      window.fetch = (url, init) => { if (init && init.method === 'POST') window.sent.push(init.body); return real(url, init); };
      const $ = id => document.getElementById(id);
      $('size').value = 'custom'; $('size').dispatchEvent(new Event('change'));
      $('width').value = '1000'; $('height').value = '384';
      $('generate').click();
    })()`);
    console.log('custom row shown:', await run(`!document.getElementById('custom-size').hidden`));
    console.log('bad width status:', await run(`document.getElementById('status').textContent`));
    console.log('requests sent for bad width:', await run(`window.sent.length`));

    await run(`(() => {
      const $ = id => document.getElementById(id);
      $('prompt').value = 'a red fox sitting in fresh snow, v3ct0r style, simple flat vector art, isolated on white bg';
      $('seed').value = '0';
      $('width').value = '640';
      for (const box of $('lora-list').querySelectorAll('input')) box.checked = box.value === 'vector';
      $('generate').click();
    })()`);
    await waitFor(`/Done|failed/.test(document.getElementById('status').textContent)`, 240000);
    console.log('status:', await run(`document.getElementById('status').textContent`));
    console.log('request body:', await run(`window.sent[window.sent.length - 1]`));
    console.log('image size:', await run(`(() => { const o = document.getElementById('output'); return o.naturalWidth + 'x' + o.naturalHeight; })()`));
    console.log('details:', await run(`[...document.querySelectorAll('#details div')].map(d => d.children[0].textContent + '=' + d.children[1].textContent).join('; ')`));
    const png = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
    fs.writeFileSync(shot, Buffer.from(png.result.data, 'base64'));
    ws.close();
  } finally {
    chrome.kill();
  }
})().catch(e => { console.error(e); process.exit(1); });
