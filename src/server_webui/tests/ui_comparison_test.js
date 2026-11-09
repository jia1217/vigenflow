// Exercise the served UI with deterministic image responses; no NPU jobs are launched.
// node ui_comparison_test.js http://127.0.0.1:11297/ /tmp/comparison-test
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const WebSocket = require('/usr/share/nodejs/ws');
const [base, artifacts] = process.argv.slice(2);
assert(base && artifacts, 'Provide the server URL and an artifact directory');
fs.mkdirSync(artifacts, { recursive: true });
const profile = fs.mkdtempSync(path.join(artifacts, 'chrome-'));
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const chrome = spawn('google-chrome', ['--headless=new', '--no-sandbox', '--disable-gpu', '--no-first-run',
    '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--window-size=1440,1100', base], { stdio: 'ignore' });
  let ws;
  try {
    const portFile = path.join(profile, 'DevToolsActivePort');
    for (let i = 0; !fs.existsSync(portFile) && i < 150; i++) await delay(100);
    const port = fs.readFileSync(portFile, 'utf8').split('\n')[0];
    let page;
    for (let i = 0; !page && i < 100; i++) {
      page = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find(t => t.type === 'page' && t.url.startsWith(base));
      if (!page) await delay(100);
    }
    assert(page, 'Browser did not open the test page');
    ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => { ws.once('open', resolve); ws.once('error', reject); });
    let id = 0;
    const pending = new Map();
    ws.on('message', raw => { const m = JSON.parse(raw); if (pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
    const send = async (method, params = {}) => {
      const result = await new Promise(resolve => { pending.set(++id, resolve); ws.send(JSON.stringify({ id, method, params })); });
      assert(!result.error, JSON.stringify(result.error));
      return result.result;
    };
    const run = async expression => {
      const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
      assert(!result.exceptionDetails, JSON.stringify(result.exceptionDetails));
      return result.result.value;
    };
    const waitFor = async expression => {
      for (let i = 0; i < 100; i++) { if (await run(expression)) return; await delay(100); }
      throw new Error('Timeout: ' + expression);
    };
    await waitFor(`document.getElementById('controls')?.disabled === false`);
    await run(`(() => {
      window.test = { requests: [], created: [], revoked: [], response: 'success' };
      const create = URL.createObjectURL.bind(URL), revoke = URL.revokeObjectURL.bind(URL);
      URL.createObjectURL = blob => { const url = create(blob); test.created.push(url); return url; };
      URL.revokeObjectURL = url => { test.revoked.push(url); revoke(url); };
      test.png = (color, width = 300, height = 300, portrait = false) => {
        const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
        const ctx = canvas.getContext('2d'); ctx.fillStyle = color; ctx.fillRect(0, 0, width, height);
        if (portrait) { ctx.fillStyle = '#ff0000'; ctx.fillRect(0, 0, width, 150); ctx.fillStyle = '#0000ff'; ctx.fillRect(0, 450, width, 150); }
        return canvas.toDataURL('image/png').split(',')[1];
      };
      test.input = (color, portrait = false, bad = false) => {
        const encoded = test.png(color, 300, portrait ? 600 : 300, portrait);
        const bytes = bad ? new Uint8Array([1, 2, 3]) : Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
        const transfer = new DataTransfer(); transfer.items.add(new File([bytes], 'reference.png', {type:'image/png'}));
        document.getElementById('input-image').files = transfer.files;
        document.getElementById('input-image').dispatchEvent(new Event('change'));
      };
      test.output = test.png('#ff00ff');
      const original = window.fetch.bind(window);
      window.fetch = async (url, options) => {
        if (options?.method === 'POST' && (String(url).endsWith('images/edits') || String(url).endsWith('images/generations'))) {
          test.requests.push({url:String(url),size:options.body instanceof FormData ? options.body.get('size') : JSON.parse(options.body).size});
          if (test.response === 'failure') return new Response(JSON.stringify({error:'Test request failed'}), {status:500});
          return new Response(JSON.stringify({data:[{b64_json:test.response === 'bad-output' ? 'AQID' : test.output}]}), {status:200});
        }
        return original(url, options);
      };
    })()`);
    const select = async model => run(`(() => { const m = document.getElementById('model'); m.value = ${JSON.stringify(model)};
      m.dispatchEvent(new Event('change')); document.getElementById('prompt').value = 'Make this image colorful.'; })()`);
    const submit = async (success = true) => {
      await run(`document.getElementById('generate').click()`);
      await waitFor(`!document.getElementById('controls').disabled`);
      const message = await run(`document.getElementById('status').textContent`);
      assert(success ? message.startsWith('Done') : message.startsWith('Image request failed'), message);
      await waitFor(`document.getElementById('output').complete && document.getElementById('output').naturalWidth > 0`);
    };
    const bounds = async () => {
      await run(`document.getElementById('image-frame').scrollIntoView({block:'center'})`);
      return run(`(() => { const frame = document.getElementById('image-frame').getBoundingClientRect();
        const image = document.getElementById('output').getBoundingClientRect();
        return {x:frame.x,y:frame.y,width:frame.width,height:frame.height,imageWidth:image.width,imageHeight:image.height,scrollX,scrollY}; })()`);
    };
    const colors = async (name, expected) => {
      const r = await bounds();
      assert(Math.abs(r.width-r.imageWidth) < 1 && Math.abs(r.height-r.imageHeight) < 1, 'Comparison must align with the output');
      const shot = await send('Page.captureScreenshot', {format:'png',captureBeyondViewport:true,
        clip:{x:r.x+r.scrollX,y:r.y+r.scrollY,width:r.width,height:r.height,scale:1}});
      fs.writeFileSync(path.join(artifacts, name+'.png'), Buffer.from(shot.data, 'base64'));
      const pixels = await run(`(async () => { const image = new Image(); image.src = ${JSON.stringify('data:image/png;base64,' + shot.data)};
        await image.decode(); const c = document.createElement('canvas'); c.width=image.naturalWidth; c.height=image.naturalHeight;
        const ctx=c.getContext('2d'); ctx.drawImage(image,0,0); return [0.25,0.75].map(x => [...ctx.getImageData(Math.floor(c.width*x),Math.floor(c.height*0.25),1,1).data].slice(0,3)); })()`);
      assert.deepEqual(pixels, expected, name);
    };
    assert(await run(`document.getElementById('comparison-tools').hidden`));
    await select('flux2-klein-4B');
    await run(`test.output=test.png('#ff00ff',600,300)`);
    await submit();
    assert(await run(`document.getElementById('comparison-tools').hidden`));
    await colors('generation-wide', [[255,0,255],[255,0,255]]);

    await select('qwen-image-2.1-turbo-edit');
    await run(`test.input('#00ff00',true); test.output=test.png('#ff00ff')`);
    await submit();
    assert(await run(`!document.getElementById('comparison-tools').hidden && document.getElementById('view-compare').getAttribute('aria-pressed') === 'true'`));
    await colors('comparison-half', [[0,255,0],[255,0,255]]);
    const originalBefore = await run(`document.getElementById('before-image').src`);
    const originalAfter = await run(`document.getElementById('output').src`);
    assert.equal(await run(`document.getElementById('download').href`), originalAfter);
    await run(`document.getElementById('view-edited').click()`);
    await colors('edited-only', [[255,0,255],[255,0,255]]);
    await run(`document.getElementById('view-compare').click(); document.getElementById('compare-position').focus()`);
    const key = async value => { await send('Input.dispatchKeyEvent',{type:'keyDown',key:value,code:value}); await send('Input.dispatchKeyEvent',{type:'keyUp',key:value,code:value}); };
    await key('Home');
    assert.equal(await run(`document.getElementById('compare-position').value`),'0');
    await colors('comparison-after', [[255,0,255],[255,0,255]]);
    await key('End');
    assert.equal(await run(`document.getElementById('compare-position').value`),'100');
    await colors('comparison-before-cropped', [[0,255,0],[0,255,0]]);
    await key('ArrowLeft');
    assert.equal(await run(`document.getElementById('compare-position').value`),'99');
    let r = await bounds();
    await run(`window.dragLog=[]; for(const type of ['pointerdown','pointermove','pointerup','gotpointercapture','lostpointercapture']) document.getElementById('image-frame').addEventListener(type,event=>dragLog.push({type,x:event.clientX,button:event.button,primary:event.isPrimary,capture:document.getElementById('image-frame').hasPointerCapture(event.pointerId),value:document.getElementById('compare-position').value}));`);
    await send('Input.dispatchMouseEvent',{type:'mousePressed',x:r.x+r.width*0.2,y:r.y+r.height/2,button:'left',buttons:1,clickCount:1});
    await send('Input.dispatchMouseEvent',{type:'mouseMoved',x:r.x+r.width*0.7,y:r.y+r.height/2,button:'left',buttons:1});
    await send('Input.dispatchMouseEvent',{type:'mouseReleased',x:r.x+r.width*0.7,y:r.y+r.height/2,button:'left'});
    const mousePosition=Number(await run(`document.getElementById('compare-position').value`));
    assert(Math.abs(mousePosition-70)<2, JSON.stringify({mousePosition,bounds:r,events:await run('dragLog')}));
    assert.equal(await run(`test.requests.length`),2, 'Comparison controls must not submit the form');

    await run(`test.input('#ffff00')`);
    assert.equal(await run(`document.getElementById('before-image').src`),originalBefore);
    await run(`test.response='failure'`);
    await submit(false);
    assert.equal(await run(`document.getElementById('before-image').src`),originalBefore);
    assert.equal(await run(`document.getElementById('output').src`),originalAfter);
    assert(!await run(`test.revoked.includes(${JSON.stringify(originalBefore)})`));
    await run(`test.response='bad-output'`);
    await submit(false);
    assert.equal(await run(`document.getElementById('before-image').src`),originalBefore);

    await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
    await send('Emulation.setTouchEmulationEnabled',{enabled:true});
    r = await bounds();
    await send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:r.x+r.width*0.2,y:r.y+r.height/2}]});
    await send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:r.x+r.width*0.75,y:r.y+r.height/2}]});
    await send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
    assert(Math.abs(Number(await run(`document.getElementById('compare-position').value`))-75)<2, 'Touch divider drag');
    assert(await run(`document.documentElement.scrollWidth <= innerWidth`), 'No horizontal overflow on mobile');
    await run(`const slider=document.getElementById('compare-position'); slider.value='50'; slider.dispatchEvent(new Event('input'))`);
    await colors('comparison-mobile', [[0,255,0],[255,0,255]]);

    await select('flux2-klein-4B-edit');
    await run(`test.response='success'; test.output=test.png('#0000ff')`);
    await submit();
    await colors('comparison-next-edit', [[255,255,0],[0,0,255]]);
    assert(await run(`test.revoked.includes(${JSON.stringify(originalBefore)}) && test.revoked.includes(${JSON.stringify(originalAfter)})`));
    await select('flux2-klein-4B');
    await submit();
    assert(await run(`document.getElementById('comparison-tools').hidden && document.getElementById('comparison-control').hidden && !document.getElementById('before-image').hasAttribute('src')`));
    await select('qwen-image-2.1-turbo-edit');
    await run(`test.input('#00ff00',false,true)`);
    await submit();
    assert(await run(`!document.getElementById('comparison-unavailable').hidden && document.getElementById('comparison-tools').hidden`));
    assert.equal(await run(`test.created.filter(url => !test.revoked.includes(url)).length`),2,'Only the current output and upload preview URLs remain');
    await send('Emulation.clearDeviceMetricsOverride');
    await send('Emulation.setTouchEmulationEnabled',{enabled:false});
    for (const model of ['flux2-klein-4B-edit','qwen-image-2.1-turbo-edit']) {
      await select(model);
      assert(await run(`Array.from(document.getElementById('size').options).some(o => o.value === 'custom')`));
      for (const [width,height] of [[768,1024],[1024,768]]) {
        await run(`(() => { const $=id=>document.getElementById(id);
          $('size').value='custom'; $('size').dispatchEvent(new Event('change'));
          $('width').value=${width}; $('height').value=${height};
          test.input('#00ff00'); test.output=test.png('#ff00ff',${width},${height}); })()`);
        await submit();
        assert.equal(await run(`test.requests.at(-1).size`),`${width}x${height}`);
        await colors(`${model}-${width}x${height}`,[[0,255,0],[255,0,255]]);
      }
      const count=await run(`test.requests.length`);
      await run(`document.getElementById('width').value='769'; document.getElementById('generate').click()`);
      assert.equal(await run(`test.requests.length`),count,'Invalid custom sizes must not submit');
    }
    console.log('PASS: both edit models, crop alignment, slider pixels, mouse/touch/keyboard, mobile layout, input pairing, failed requests, generation mode, decode fallback, URL cleanup.');
  } finally {
    if (ws) ws.close();
    chrome.kill();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
