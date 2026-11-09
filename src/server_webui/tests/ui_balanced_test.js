// Drive the served page in headless Chrome: no memory-mode selector, and generation posts no mode.
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
    const models = await (await fetch(base + 'v1/models')).json();
    for (const model of models.data) {
      await run(`(() => {
        const $ = id => document.getElementById(id);
        $('model').value = ${JSON.stringify(model.id)};
        $('model').dispatchEvent(new Event('change'));
        $('advanced-settings').open = true;
        if ($('mode') || $('mode-field')) throw new Error('Memory mode selector still present');
        if (/memory mode|performance/i.test($('advanced-settings').innerText)) throw new Error('Mode text still shown');
      })()`);
    }
    console.log('PASS: no memory mode selector for any of', models.data.length, 'models');
    await run(`(() => {
      const $ = id => document.getElementById(id);
      $('model').value = 'flux1-schnell'; $('model').dispatchEvent(new Event('change'));
      $('size').value = 'custom'; $('size').dispatchEvent(new Event('change'));
      $('width').value = $('height').value = '512';
      $('prompt').value = 'A red fox in fresh snow'; $('seed').value = '0';
      window.sent = [];
      const real = window.fetch;
      window.fetch = async (url, init) => {
        if (init && init.method === 'POST') window.sent.push(JSON.parse(init.body));
        return real(url, init);
      };
      $('generate').click();
    })()`);
    await waitFor(`!document.getElementById('controls').disabled`, 60000);
    const status = await run(`document.getElementById('status').textContent`);
    if (!status.startsWith('Done')) throw new Error(status);
    const sent = await run('window.sent');
    if (sent.length !== 1 || 'mode' in sent[0]) throw new Error('Unexpected request: ' + JSON.stringify(sent));
    const details = await run(`document.body.innerText`);
    if (/memory mode/i.test(details)) throw new Error('Memory mode still shown in details');
    console.log('PASS:', status, 'request:', JSON.stringify(sent[0]));
    const png = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
    fs.writeFileSync(shot, Buffer.from(png.result.data, 'base64'));
    ws.close();
  } finally {
    chrome.kill();
  }
})().catch(e => { console.error(e); process.exit(1); });
