#pragma once

namespace server {

// Embedded so the browser UI travels with vgf-serve on Ubuntu and Windows.
// No web server, JavaScript build, or external assets are required.
inline constexpr char kWebUi[] = R"VGFHTML(<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark light">
  <title>VigenFlow</title>
  <script>
try { document.documentElement.dataset.theme = localStorage.getItem('vgf-theme'); } catch {}
</script><style>
:root {
  color-scheme: dark;
  --bg: light-dark(#f4f5f9,#101116);
  --panel: light-dark(#fff,#181a23);
  --field: light-dark(#f8f9fc,#11131a);
  --ink: light-dark(#23243a,#f1f1fa);
  --muted: light-dark(#565b71,#a8abbd);
  --line: light-dark(#d5d8e5,#303342);
  --accent: light-dark(#494ab0,#bbbcff);
  --accent-ink: light-dark(#fff,#181638);
  --error: light-dark(#9c262b,#ffaaa7);
  font: 14px/1.5 Inter, "Segoe UI", system-ui, sans-serif;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); }
main { max-width: 1552px; margin: auto; padding: 0 32px 28px; }
.app-header, .brand, .preview-heading, .panel-heading, .composer-label, .composer-footer {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
}
.app-header { height: 76px; border-bottom: 1px solid var(--line); margin-bottom: 22px; }
.brand { justify-content: flex-start; }
.brand-mark {
  width: 34px; height: 34px; display: grid; place-items: center; border-radius: 10px;
  background: var(--accent); color: var(--accent-ink); font-size: 21px; font-weight: 850;
}
h1, h2, p { margin: 0; }
h1 { font-size: 20px; letter-spacing: -.45px; }
h2, legend { font-size: 15px; font-weight: 700; }
.badge {
  display: inline-flex; align-items: center; gap: 7px; padding: 4px 10px; border-radius: 20px;
  background: light-dark(#edf0f7,#232533); color: var(--muted); font-size: 11px; font-weight: 650;
}
#connection-state::before { content: ''; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
#connection-state[data-state="connected"] { color: light-dark(#205d43,#a0e6c2); background: light-dark(#e1f4e9,#1b2d2a); }
#connection-state[data-state="error"] { color: var(--error); background: light-dark(#fce7eb,#38252c); }
.workspace { display: grid; grid-template-columns: 326px minmax(0, 1fr); gap: 20px; align-items: start; }
.panel { min-width: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 16px; }
.settings { padding: 18px; }
.panel-heading { margin-bottom: 16px; }
.field { margin-bottom: 14px; min-width: 0; }
label { display: block; margin-bottom: 6px; font-size: 12px; font-weight: 650; }
fieldset { margin: 0; padding: 0; border: 0; min-width: 0; }
legend { padding: 0; }
input, textarea, select, button, .download, .secondary-link {
  max-width: 100%; padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px;
  background: var(--field); color: inherit; font: inherit;
}
input, textarea, select { width: 100%; min-width: 0; }
input, select, button, .download, .secondary-link { min-height: 36px; }
textarea { display: block; min-height: 74px; resize: vertical; line-height: 1.55; }
textarea::placeholder { color: light-dark(#656b81,#9396ab); }
input[readonly], select:disabled { color: var(--muted); }
input[type="checkbox"] { width: 16px; height: 16px; min-height: 0; margin: 0; flex: none; accent-color: var(--accent); }
button, .download, .secondary-link {
  cursor: pointer; text-align: center; font-weight: 650; font-size: 12px; text-decoration: none;
}
button:hover:not(:disabled), .secondary-link:hover { border-color: light-dark(#777eaa,#6e7289); background: light-dark(#f0f0fb,#262937); }
button:disabled { opacity: .55; cursor: not-allowed; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.help { display: block; margin-top: 6px; color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
.section { margin: 16px 0 0; padding-top: 14px; border-top: 1px solid var(--line); }
.section h2 { margin-bottom: 10px; }
.seed-row { display: flex; gap: 7px; }
.seed-row button { flex: none; }
.grid, .custom-size { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.custom-size { margin-top: 8px; }
.custom-size label { color: var(--muted); }
summary { cursor: pointer; font-weight: 650; font-size: 12px; padding: 11px 0; }
.fold { border-top: 1px solid var(--line); }
.fold > summary { color: var(--ink); }
.fold > summary::marker, .result-details > summary::marker { color: var(--accent); }
.fold > .field:last-child { margin-bottom: 8px; }
#refresh { min-height: 30px; padding: 3px 9px; }
#retry { width: 100%; margin-top: 10px; }
.ref-card {
  display:grid; grid-template-columns:56px minmax(0,1fr); gap:8px; align-items:center;
  margin-top:8px; padding:8px; border:1px solid var(--line); border-radius:10px;
}
.ref-card label { margin:0; }
.ref-heading { display:flex; align-items:center; justify-content:space-between; gap:4px; margin-bottom:3px; }
.ref-fields { --ref-field-height:32px; display:grid; grid-template-columns:minmax(0,1fr) auto auto; gap:4px; }
.ref-fields :is(input,button) { height:var(--ref-field-height); min-height:0; }
.ref-fields button { width:var(--ref-field-height); padding:0; }
.ref-drag { min-height:0; padding:0 5px; cursor:grab; color:var(--muted); line-height:1.3; user-select:none; }
.ref-drag:active { cursor:grabbing; }
.ref-card[data-dragging] { opacity:.5; }
.ref-card[data-drop-position="before"] { box-shadow:0 -2px var(--accent); }
.ref-card[data-drop-position="after"] { box-shadow:0 2px var(--accent); }
#reference-budget[data-over-limit] { color:var(--error); }
.ref-name { padding:4px 6px; }
.ref-card .help { margin-top:3px; }
.open-reference { width:56px; height:56px; padding:0; overflow:hidden; cursor:zoom-in; }
.input-preview { display:block; width:100%; height:100%; object-fit:contain; }
input[type="file"] { font-size: 11px; }
input::file-selector-button {
  margin-right: 7px; padding: 4px 7px; border: 1px solid var(--line); border-radius: 5px;
  background: light-dark(#e9ebf5,#242637); color: var(--ink); cursor: pointer;
}
.studio { min-width: 0; display: grid; gap: 14px; }
.preview-panel { padding: 14px; scroll-margin-top: 10px; }
.preview-heading { margin: 0 3px 12px; }
.preview-placeholder, .image-stage {
  display: grid; place-items: center; min-height: clamp(220px, 40vh, 420px); border-radius: 10px;
  border: 1px solid light-dark(#d8dbee,#262936); background-color: light-dark(#f8f9fc,#11131a);
  background-image: radial-gradient(ellipse at center, light-dark(#e8e8fa,#252744) 0%, transparent 68%), radial-gradient(light-dark(#c4c8e4,#393d54) 1px, transparent 1px);
  background-size: 100% 100%, 21px 21px;
}
.preview-placeholder { padding: 24px; text-align: center; color: var(--muted); }
.empty-state { display: grid; justify-items: center; gap: 7px; }
.empty-state svg {
  width: 49px; height: 49px; padding: 9px; border: 1px solid light-dark(#858bc0,#555981);
  border-radius: 12px; background: light-dark(#ececff,#22243a); color: var(--accent); margin-bottom: 6px;
}
.empty-state strong { font-size: 16px; color: var(--ink); }
.empty-state p { max-width: 270px; font-size: 12px; }
.preview-panel[data-busy="true"] #preview-state { background: light-dark(#e6e8fa,#2a2b49); color: var(--accent); }
.preview-panel[data-busy="true"] #preview-state::before {
  content: ''; width: 9px; height: 9px; border: 2px solid light-dark(#b2b7dd,#646890); border-top-color: var(--accent);
  border-radius: 50%; animation: spin .9s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.image-stage { padding: 10px; min-height: 0; }
#output { display: block; width: 100%; height: auto; }
.image-frame { position: relative; width: 100%; max-width: calc(46vh * var(--image-ratio, 1)); overflow: hidden; border-radius: 5px; --split: 50%; }
.image-frame[data-comparing="true"] { cursor: ew-resize; touch-action: pan-y; user-select: none; }
.compare-before { position: absolute; inset: 0; clip-path: inset(0 calc(100% - var(--split)) 0 0); pointer-events: none; }
#before-image { display: block; width: 100%; height: 100%; object-fit: cover; object-position: center; }
.compare-divider {
  position: absolute; inset: 0 auto 0 var(--split); width: 2px; transform: translateX(-50%);
  background: white; box-shadow: 0 0 4px #151627aa; pointer-events: none;
}
.compare-divider span {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); display: grid; place-items: center;
  width: 33px; height: 33px; border: 2px solid white; border-radius: 50%;
  background: var(--accent); color: var(--accent-ink); font-size: 17px;
}
.comparison-label {
  position: absolute; top: 9px; padding: 3px 8px; border-radius: 6px;
  background: #11131add; color: white; font-size: 11px; pointer-events: none;
}
#before-label { left: 9px; }
#after-label { right: 9px; }
.result-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 8px; }
.view-switch { display: inline-flex; gap: 3px; padding: 3px; background: var(--field); border: 1px solid var(--line); border-radius: 8px; }
.view-switch button { min-height: 30px; padding: 4px 8px; border-color: transparent; color: var(--muted); }
.view-switch button[aria-pressed="true"] { background: light-dark(#e7e9fb,#292c42); color: var(--accent); }
.comparison-control { margin-top: 8px; }
#compare-position { display: block; min-height: 24px; padding: 0; border: 0; accent-color: var(--accent); cursor: ew-resize; }
.result-toolbar { display: flex; flex-wrap: wrap; gap: 8px; margin-left: auto; }
.download, .secondary-link { display: inline-flex; align-items: center; justify-content: center; }
#result-summary { margin: 9px 0 0; color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
.result-details { margin-top: 8px; border-top: 1px solid var(--line); }
#details { margin: 0; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
#details dt { color: var(--muted); font-size: 11px; }
#details dd { margin: 2px 0 0; }
#details > div, #output-prompt { min-width: 0; overflow-wrap: anywhere; }
#output-prompt { margin-top: 10px; font-size: 12px; white-space: pre-wrap; }
.composer { padding: 13px 14px; }
.composer-label { margin-bottom: 6px; }
.composer-label label { font-size: 13px; margin: 0; }
.composer-label .help { margin: 0; }
.composer-footer { margin-top: 9px; }
#status { color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
#status[data-error="true"], #lora-error { color: var(--error); }
#generate, .download { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
#generate { min-width: 132px; min-height: 38px; flex: none; }
#generate:hover:not(:disabled), .download:hover { background: light-dark(#393a98,#d2d2ff); }
#generate:disabled { cursor: not-allowed; }
#lora-list { display: grid; gap: 6px; }
.lora { position: relative; border: 1px solid var(--line); border-radius: 8px; padding: 6px 8px; background: var(--field); }
.lora:has(input:checked) { border-color: light-dark(#7479c2,#8488d7); background: light-dark(#ededff,#242641); }
.lora[data-missing="true"] { color: var(--muted); }
.lora-row { display: flex; align-items: center; gap: 6px; padding-right: 52px; }
.lora label { display: flex; flex: 1; min-width: 0; min-height: 30px; align-items: center; gap: 7px; margin: 0; cursor: pointer; }
.lora-name { min-width: 0; overflow-wrap: anywhere; }
.lora .badge { font-size: 10px; padding: 2px 6px; margin-bottom: 5px; }
.lora-trigger { margin: 0 0 5px; font-size: 11px; color: var(--muted); overflow-wrap: anywhere; }
.lora-add { flex: none; padding: 4px 5px; min-height: 27px; font-size: 10px; color: var(--accent); }
.lora summary { position: absolute; top: 6px; right: 8px; padding: 6px 0; font-size: 10px; color: var(--muted); }
.lora-info { margin-top: 5px; padding-top: 7px; border-top: 1px solid var(--line); }
.lora small { display: block; font-size: 10px; color: var(--muted); overflow-wrap: anywhere; }
[hidden] { display: none !important; }
@media (max-width: 760px) {
  main { padding: 0 14px 20px; }
  .app-header { height: 59px; margin-bottom: 12px; }
  .workspace, .studio { grid-template-columns: minmax(0, 1fr); gap: 12px; }
  .settings { padding: 13px; }
  .image-options:not(:has(#custom-size:not([hidden]))) {
    display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); gap: 9px;
  }
  .preview-placeholder { min-height: 190px; }
}
@media (max-width: 400px) {
  .composer-label, .composer-footer { flex-wrap: wrap; }
  #generate { width: 100%; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before { animation: none !important; scroll-behavior: auto !important; }
}
  .ratio-toggle{display:flex;align-items:center;gap:7px;margin-top:8px}
.prompt-nav { display:flex; gap:5px; }
.prompt-nav a[aria-current="page"] { color:var(--accent); border-color:light-dark(#858bc0,#555981); }
.prompt-library { max-width:760px; margin:auto; padding:18px; }
#prompt-editor { display:grid; gap:10px; margin-top:14px; }
#prompt-editor textarea { min-height:130px; }
.prompt-row { padding:12px 0; border-bottom:1px solid var(--line); }
.prompt-row p { white-space:pre-wrap; overflow-wrap:anywhere; margin:6px 0; }
.prompt-row strong, .prompt-row code { overflow-wrap:anywhere; }
.prompt-actions { display:flex; gap:6px; flex-wrap:wrap; }
.prompt-picker { margin:5px 0; max-height:180px; overflow:auto; border:1px solid var(--line); border-radius:8px; padding:4px; }
.prompt-picker button { display:block; width:100%; text-align:left; border:0; overflow-wrap:anywhere; }
.prompt-picker button[aria-selected="true"] { background:light-dark(#e7e9fb,#292c42); color:var(--accent); }
#prompt-fields,#preset-dialog { width:min(440px,calc(100% - 28px)); max-height:85vh; overflow:auto; color:var(--ink); padding:18px; }
#prompt-fields::backdrop,#preset-dialog::backdrop { background:#000a; }
#prompt-fields-title { overflow-wrap:anywhere; }
#variable-fields { margin:14px 0; }
#variable-fields input,#variable-fields textarea { margin:4px 0 8px; }
@media (max-width:500px) {
  .app-header { height:auto; flex-wrap:wrap; padding:10px 0; }
  .prompt-nav { order:3; width:100%; }
}

.settings { position:relative; }
#drop-hint {
  position:absolute; inset:0; z-index:2; pointer-events:none; border-radius:inherit;
  display:grid; place-content:center; justify-items:center; gap:12px; text-align:center;
  border:2px dashed var(--accent); color:var(--ink); background:light-dark(#fff,#181a23); overflow:hidden;
}
#drop-hint::before {
  content:''; position:absolute; inset:-18px;
  background:radial-gradient(circle,#7377b8 1px,transparent 2px) 0 0 / 18px 18px;
  opacity:.35; animation:drop-dots 1.8s linear infinite;
}
@keyframes drop-dots { to { transform:translate(18px,18px); } }
#drop-hint span, #drop-hint strong { position:relative; background:var(--panel); padding:4px 10px; }
#drop-hint span { font-size:28px; color:var(--accent); border:1px solid var(--accent); border-radius:10px; }

:root[data-theme=light]{color-scheme:light}
.header-status{display:flex;gap:8px}
#theme-toggle{width:42px;min-height:0;padding:2px;border-radius:20px;background:var(--line)}
#theme-toggle::before{content:'☾';display:block;width:16px;height:16px;border-radius:50%;background:var(--ink);color:var(--bg);font:12px/16px sans-serif;transition:transform .18s}
#theme-toggle[aria-checked=true]{background:var(--accent)}
#theme-toggle[aria-checked=true]::before{content:"☀";transform:translateX(20px);background:#fff;color:var(--accent)}

#prompt[contenteditable]{display:block;min-height:74px;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--field);white-space:pre-wrap;overflow-wrap:anywhere}
#prompt:empty::before{content:attr(data-placeholder);color:var(--muted);pointer-events:none}
.ref-chip{display:inline-flex;align-items:center;gap:4px;vertical-align:middle;padding:1px 6px;border:1px solid var(--line);border-radius:7px;font-size:0;background:var(--field);max-width:100%;user-select:all}
.ref-chip img{width:19px;height:19px;object-fit:cover;border-radius:3px}
.ref-chip::after{content:attr(data-label);font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ref-chip[data-missing]{color:var(--error)}
.ref-chip[data-missing] img{display:none}
body:has(#reference-view[open]) { overflow:hidden; }
#reference-view {
  inset:0; width:100%; height:100dvh; max-width:none; max-height:none; margin:0;
  padding:clamp(12px,3vw,32px); border:0; color:var(--ink); background:var(--bg);
}
#reference-view[open] { display:grid; grid-template-rows:auto minmax(0,1fr); gap:14px; }
.reference-toolbar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:flex-end; }
#reference-view[open]:has(#viewer-prompt:not([hidden])) { grid-template-rows:auto minmax(0,1fr) auto; }
#viewer-prompt { width:min(100%,960px); justify-self:center; padding:12px; }
#viewer-used-prompt { max-height:20dvh; padding-right:5px; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere; }
.reference-toolbar > div { flex:1 1 130px; min-width:0; overflow-wrap:anywhere; }
#image-frame:not([data-comparing="true"]) #output { cursor:zoom-in; }
#full-reference { width:100%; height:100%; min-height:0; object-fit:contain; }
#input-limit { margin-top:8px; }
#input-limit legend { margin-bottom:6px; font-size:12px; }
.size-track {
  display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:2px; padding:3px;
  border:1px solid var(--line); border-radius:24px; background:var(--field);
}
.size-track:has(label:nth-child(4):not([hidden]), label:nth-child(5):not([hidden])) { grid-template-columns:repeat(4,minmax(0,1fr)); }
.size-track label { position:relative; margin:0; min-width:0; cursor:pointer; }
.size-track input { position:absolute; width:1px; height:1px; min-height:0; opacity:0; }
.size-track span {
  display:grid; place-items:center; min-height:30px; padding:1px 4px; border-radius:18px;
  color:var(--muted); font-variant-numeric:tabular-nums;
}
.size-track label:hover span { color:var(--ink); }
.size-track label:has(input:disabled) { cursor:not-allowed; }
.size-track input:disabled + span { opacity:.42; color:var(--muted); }
.size-track input:checked + span { background:var(--accent); color:var(--accent-ink); }
.size-track input:focus-visible + span { outline:2px solid var(--accent); outline-offset:2px; }

.settings { padding:10px 12px; }
.settings .panel-heading { margin-bottom:6px; }
.settings .panel-heading label { margin:0; }
.settings .field { margin-bottom:10px; }
.settings .section { margin-top:11px; padding-top:10px; }
.settings summary { padding:8px 0; }
.settings .fold .field { margin-bottom:10px; }
.ref-header,.output-heading { display:flex; align-items:center; justify-content:space-between; gap:6px; }
#setup-heading,.ref-header h2,#input-limit legend,.output-heading h2 { font-size:12px; font-weight:700; }
.ref-header h2,.output-heading h2 { margin:0; }
.ref-upload {
  position:relative; overflow:hidden; margin:0; padding:4px 9px;
  border:1px solid var(--line); border-radius:7px; background:var(--field); cursor:pointer;
}
.ref-upload:hover { border-color:var(--muted); }
.ref-upload:focus-within { outline:2px solid var(--accent); outline-offset:2px; }
#input-image { position:absolute; inset:0; width:100%; height:100%; opacity:0; padding:0; cursor:pointer; }
.ref-card { margin-top:6px; }
#reference-size { position:relative; }
#input-limit { margin-top:7px; }
#input-limit .size-track span { min-height:26px; }
#input-upscaling-note { margin-top:3px; }
#edit-size.section { margin-top:5px; padding-top:6px; }
.sending-details { position:absolute; top:0; right:0; display:flex; align-items:center; gap:5px; }
#sending-size { margin:0; }
#sending-info { width:17px; height:17px; min-height:0; padding:0; border-radius:50%; color:var(--muted); font-size:11px; font-weight:700; line-height:1; }
#sending-tooltip {
  position:absolute; z-index:5; top:100%; right:0; width:min(242px, calc(100vw - 48px));
  padding:9px 11px; border:1px solid var(--line); border-radius:8px;
  background:var(--panel); color:var(--ink); box-shadow:0 5px 20px #0003;
  font-size:12px; line-height:1.4; visibility:hidden;
}
.sending-details:not([data-dismissed]) #sending-info:is(:hover,:focus) + #sending-tooltip { visibility:visible; }
#edit-size .output-heading { margin-bottom:6px; }
#edit-size-help { margin:0; }
#edit-size input:not([type="checkbox"]),#edit-preset { min-height:32px; padding-block:5px; }
.output-toggles { display:flex; flex-wrap:wrap; gap:0 8px; }
.output-toggles label { font-size:11px; margin-bottom:0; }
</style>
</head>
<body>
<main>
  <header class="app-header">
    <div class="brand">
      <span class="brand-mark" aria-hidden="true">V</span>
      <h1>VigenFlow</h1>
    </div>
    <nav class="prompt-nav" aria-label="Workspace">
      <a id="studio-link" class="secondary-link" href="#studio">Studio</a>
      <a id="prompts-link" class="secondary-link" href="#prompts">Prompts</a>
    </nav>
    <div class="header-status"><button id="theme-toggle" type="button" role="switch" aria-label="Light theme" aria-checked="false"></button><span id="connection-state" class="badge" role="status" data-state="connecting">Connecting</span></div>
  </header>
  <noscript>This image client needs JavaScript enabled in your browser.</noscript>
  <form id="generation-form" class="workspace">
    <section class="settings panel" aria-labelledby="setup-heading">
      <fieldset id="controls" disabled>
        <div class="panel-heading">
          <label id="setup-heading" for="model">Model</label>
          <button id="refresh" type="button" aria-label="Refresh models and styles">Refresh</button>
        </div>
        <div class="field">
          <select id="model" required><option value="">Loading models…</option></select>
        </div>
            <div id="input-field" class="field" hidden>
              <div class="ref-header">
                <h2>References</h2>
                <label class="ref-upload"><span id="ref-upload-label">Add image</span><input id="input-image" type="file" accept="image/png,image/jpeg,image/bmp,image/gif" aria-describedby="input-help"></label>
              </div>
              <small id="input-help" class="help">Drop or paste an image. PNG, JPEG, BMP or GIF, up to 40 MB.</small>
              <div id="reference-list"></div>

              <div id="reference-size" hidden>
                <div class="sending-details">
                  <p id="sending-size" class="help" role="status"></p>
                  <button id="sending-info" type="button" aria-label="About input dimensions" aria-describedby="sending-tooltip">i</button>
                  <span id="sending-tooltip" role="tooltip">Edit sizes use 16 px increments. Match input uses the first reference to set the output size.</span>
                </div>
                <fieldset id="input-limit" aria-describedby="input-size-note">
                  <legend>Input limit</legend>
                  <div class="size-track">
                    <label><input type="radio" name="reference-limit" value="512"><span>512</span></label>
                    <label><input type="radio" name="reference-limit" value="1024"><span>1024</span></label>
                    <label><input type="radio" name="reference-limit" value="2048"><span>2048</span></label>
                    <label><input type="radio" name="reference-limit" value="max" checked><span>Auto</span></label>
                  </div>
                </fieldset>
                <small id="reference-budget" class="help" role="status" aria-describedby="reference-budget-help" hidden></small>
                <small id="reference-budget-help" class="help" hidden>Each 16 × 16 block is one token. Auto fits the budget by reducing larger images first. Smaller images keep their size until needed.</small>
                <small id="input-size-note" hidden>Long-edge limits only reduce larger images. Auto uses up to 2048 px and fits the reference token budget.</small>
                <small id="input-upscaling-note" class="help">Upscaling for input pictures coming soon.</small>
              </div>
              <div id="edit-size" class="section" hidden>
                <div class="output-heading"><h2>Output size</h2><small id="edit-size-help" class="help"></small></div>
                <select id="edit-preset" aria-label="Size preset"></select>
                <div id="edit-dimensions"></div>
                <div class="output-toggles">
                  <label class="ratio-toggle">
                  <input id="lock-ratio" type="checkbox" aria-label="Keep input aspect ratio" checked> Keep ratio
                </label>
                <label class="ratio-toggle"><input id="source-size" type="checkbox" aria-label="Match output to input"> Match input</label>
                </div>
              </div>
            </div>
            <section id="generation-size" class="section" aria-label="Image size">
                <div class="field">
                  <label for="size">Image size</label>
                  <select id="size" aria-describedby="size-help"><option value="1024x1024">1024 × 1024</option></select>
                  <div id="generation-dimensions"><div id="custom-size" class="custom-size" hidden>
                    <div><label for="width">Width (px)</label><input id="width" type="number" value="1024" required disabled></div>
                    <div><label for="height">Height (px)</label><input id="height" type="number" value="1024" required disabled></div>
                  </div>
                  </div>
                  <small id="size-help" class="help">Sizes supported by this model.</small>
                </div>
            </section>
            <details id="lora-field" class="fold" hidden>
              <summary>Styles (LoRAs) <span id="style-count"></span></summary>
              <div id="lora-list"></div>
              <p class="help field">Add words also selects the style.</p>
              <small id="lora-error" class="help field" hidden></small>
            </details>
            <details id="advanced-settings" class="fold">
              <summary>Advanced settings</summary>
                <div class="field">
                  <label for="seed">Seed</label>
                  <div class="seed-row">
                    <input id="seed" type="number" min="-2147483648" max="2147483647" step="1" value="42" required aria-describedby="seed-help">
                    <button id="random-seed" type="button" aria-label="Choose a random seed">Random</button>
                  </div>
                  <small id="seed-help" class="help">Reuse a seed to repeat a result.</small>
                </div>

              <div class="grid">
                <div class="field">
                  <label for="steps">Generation steps</label>
                  <select id="steps" required aria-describedby="steps-help"></select>
                  <small id="steps-help" class="help">Fixed for this model.</small>
                </div>
                <div class="field">
                  <label for="response-format">Image delivery</label>
                  <select id="response-format"><option value="url">URL (default)</option><option value="b64_json">Base64</option></select>

                </div>
              </div>

              <div class="field">
                <label for="base-url">Server address</label>
                <input id="base-url" type="url" readonly>

              </div>
            </details>
      </fieldset>
      <div id="drop-hint" aria-hidden="true" hidden><span>+</span><strong>Drop image here</strong></div>
      <button id="retry" type="button" hidden>Retry connection</button>
    </section>
    <div class="studio">
      <section id="preview-panel" class="preview-panel panel" aria-labelledby="result-heading">
      <div class="preview-heading">
        <h2 id="result-heading">Preview</h2>
        <span id="preview-state" class="badge">Waiting for input</span>
      </div>
      <div id="preview-placeholder" class="preview-placeholder">
        <div class="empty-state">
        <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><rect x="5" y="5" width="38" height="38" rx="7"/><circle cx="16" cy="16" r="4"/><path d="M6 35l11-11 8 8 6-6 12 12"/></svg>
        <strong id="preview-message">No image yet</strong>
        <p id="preview-hint">Enter a prompt to start.</p>
        </div>
      </div>
      <div id="result" hidden>
        <div class="result-actions">
        <div id="comparison-tools" class="comparison-tools" hidden>
          <div class="view-switch" role="group" aria-label="Image view">
            <button id="view-edited" type="button" aria-pressed="true">Edited image</button>
            <button id="view-compare" type="button" aria-pressed="false">Before / after</button>
          </div>
        </div>
        <div class="result-toolbar">
          <a id="download" class="download" download="vigenflow.png">Download PNG</a>
          <a id="open-image" class="secondary-link" target="_blank" rel="noopener">View full size</a>
        </div>
        </div>
        <div class="image-stage">
          <div id="image-frame" class="image-frame">
            <img id="output" alt="Generated image" draggable="false">
            <div id="compare-before" class="compare-before" hidden><img id="before-image" alt="Original image cropped to match" draggable="false"></div>
            <div id="compare-divider" class="compare-divider" aria-hidden="true" hidden><span>↔</span></div>
            <span id="before-label" class="comparison-label" aria-hidden="true" hidden>Before</span>
            <span id="after-label" class="comparison-label" aria-hidden="true" hidden>After</span>
          </div>
        </div>
        <div id="comparison-control" class="comparison-control" hidden>
          <label for="compare-position">Slide to compare</label>
          <input id="compare-position" type="range" min="0" max="100" value="50" aria-describedby="comparison-help">
          <small id="comparison-help" class="help">Drag the divider or use the arrow keys.</small>
        </div>
        <p id="comparison-unavailable" class="help" hidden>Original image unavailable.</p>
        <p id="result-summary"></p>
        <details class="result-details">
          <summary>Generation details</summary>
          <dl id="details"></dl>
          <p id="output-prompt"></p>
        </details>
      </div>
      </section>
      <section class="composer panel" aria-labelledby="prompt-label">
        <fieldset id="composer-controls" disabled>
          <div class="composer-label">
            <label id="prompt-label" for="prompt">Prompt</label>
            <small id="prompt-help" class="help">/ Presets · @ Image · Ctrl / ⌘ + Enter</small>
          </div>
          <div id="prompt" role="textbox" aria-multiline="true" aria-labelledby="prompt-label" aria-describedby="prompt-help" aria-autocomplete="list" aria-controls="prompt-picker" data-placeholder="Describe your image…" contenteditable="false"></div>
          <div id="prompt-picker" class="prompt-picker" role="listbox" aria-label="Prompt suggestions" hidden></div>
          <div class="composer-footer">
            <button id="save-preset" type="button">Save as preset</button>
            <div id="status" role="status" aria-live="polite">Connecting to your server…</div>
            <button id="generate" type="submit">Generate image</button>
          </div>
        </fieldset>
      </section>
    </div>
  </form>

<section id="prompt-library" class="panel prompt-library" aria-labelledby="library-heading" hidden>
  <div class="panel-heading">
    <h2 id="library-heading">Saved prompts</h2>
    <button id="new-prompt" type="button">New prompt</button>
  </div>
  <div id="prompt-list"></div>
  <p id="library-status" class="help" role="status"></p>
</section>
<dialog id="preset-dialog" class="panel" aria-labelledby="preset-title">
  <h2 id="preset-title">New prompt</h2>
  <form id="prompt-editor">
    <div><label for="template-name">Name</label><input id="template-name" required maxlength="100"></div>
    <div><label for="template-command">Slash command</label><input id="template-command" placeholder="/portrait" required maxlength="65" autocomplete="off"></div>
    <div><label for="template-content">Prompt content</label><textarea id="template-content" required maxlength="16384"></textarea>
      <small class="help">Use <code>{{Who is in the photo}}</code> to ask for a value. <code>| text:required</code> requires a value. <code>| textarea</code> allows multiple lines.</small></div>
    <div class="prompt-actions">
      <button type="submit">Save prompt</button>
      <button id="cancel-prompt" type="button">Cancel</button>
    </div>
    <p id="preset-status" class="help" role="status"></p>
  </form>
</dialog>

<dialog id="prompt-fields" class="panel" aria-labelledby="prompt-fields-title">
  <form id="prompt-fields-form">
    <h2 id="prompt-fields-title"></h2>
    <div id="variable-fields"></div>
    <div class="prompt-actions">
      <button type="submit">Insert prompt</button>
      <button id="cancel-fields" type="button">Cancel</button>
    </div>
  </form>
</dialog>

<dialog id="reference-view" aria-label="Image preview">
  <header class="reference-toolbar">
    <div id="reference-info" hidden><h2 id="reference-title"></h2><p id="reference-original" class="help"></p></div>
    <a id="download-reference" class="secondary-link" download>Download original</a>
    <button id="close-reference" type="button">Close</button>
  </header>
  <img id="full-reference" alt="">
  <section id="viewer-prompt" class="panel" aria-labelledby="viewer-prompt-title" hidden>
    <h2 id="viewer-prompt-title">Prompt</h2>
    <p id="viewer-used-prompt" tabindex="0"></p>
  </section>
</dialog>
</main>
<script>
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const api = new URL('/v1/', location.origin);
  $('base-url').value = api.href.replace(/\/$/, '');
  let models = [];
  let outputUrl = '';
  let beforeUrl = '';
  let comparing = false;

  function updateComparisonPosition(value) {
    const position = Math.max(0, Math.min(100, Number(value)));
    $('compare-position').value = String(position);
    $('compare-position').setAttribute('aria-valuetext', `${Math.round(position)}% before, ${Math.round(100 - position)}% after`);
    $('image-frame').style.setProperty('--split', `${position}%`);
    $('before-label').hidden = !comparing || position < 5;
    $('after-label').hidden = !comparing || position > 95;
  }

  function showComparison(enabled) {
    comparing = Boolean(enabled && beforeUrl);
    $('image-frame').dataset.comparing = String(comparing);
    for (const id of ['compare-before', 'compare-divider', 'comparison-control']) $(id).hidden = !comparing;
    $('view-edited').setAttribute('aria-pressed', String(!comparing));
    $('view-compare').setAttribute('aria-pressed', String(comparing));
    updateComparisonPosition($('compare-position').value);
  }

  async function comparisonInput(file) {
    if (!file) return '';
    const url = URL.createObjectURL(file);
    try {
      const input = new Image();
      input.src = url;
      await input.decode();
      return url;
    } catch {
      URL.revokeObjectURL(url);
      return '';
    }
  }

  $('view-edited').addEventListener('click', () => showComparison(false));
  $('view-compare').addEventListener('click', () => showComparison(true));
  $('compare-position').addEventListener('input', event => updateComparisonPosition(event.target.value));
  const moveDivider = event => {
    const bounds = $('image-frame').getBoundingClientRect();
    if (bounds.width) updateComparisonPosition((event.clientX - bounds.left) / bounds.width * 100);
  };
  $('image-frame').addEventListener('pointerdown', event => {
    if (!comparing || !event.isPrimary || event.button !== 0) return;
    event.preventDefault();
    $('compare-position').focus({ preventScroll: true });
    $('image-frame').setPointerCapture(event.pointerId);
    moveDivider(event);
  });
  $('image-frame').addEventListener('pointermove', event => {
    if (comparing && $('image-frame').hasPointerCapture(event.pointerId)) moveDivider(event);
  });
  $('image-frame').addEventListener('pointerup', event => {
    if ($('image-frame').hasPointerCapture(event.pointerId)) $('image-frame').releasePointerCapture(event.pointerId);
  });

  function setConnection(state, label) {
    $('connection-state').dataset.state = state;
    $('connection-state').textContent = label;
  }

  function updateAction(model) {
    $('generate').textContent = model.edits_image ? 'Edit image' : 'Generate image';
    $('prompt-label').textContent = model.edits_image ? 'Editing instructions' : 'Prompt';
    $('prompt').placeholder = model.edits_image
      ? 'Describe what you want to change…'
      : 'Describe your image…';
    if (!outputUrl) {
      $('preview-state').textContent = 'Waiting for input';
      $('preview-message').textContent = model.edits_image ? 'No edited image yet' : 'No image yet';
      $('preview-hint').textContent = model.edits_image
        ? 'Add an image and a prompt.'
        : 'Enter a prompt to start.';
    }
  }

  function disableControls(disabled) {
    $('controls').disabled = $('composer-controls').disabled = $('prompt').disabled = disabled;
  }

  function status(message, error = false) {
    $('status').textContent = message;
    $('status').dataset.error = String(error);
  }

  async function readResponse(response) {
    let data;
    try { data = await response.json(); }
    catch { throw new Error(`Invalid response (HTTP ${response.status}).`); }
    if (!response.ok) {
      const error = typeof data.error === 'string' ? data.error : data.error?.message;
      throw new Error(error || `Request failed (HTTP ${response.status}).`);
    }
    return data;
  }

  function fillOptions(select, values, selected) {
    select.replaceChildren(...values.map(value => new Option(String(value), String(value))));
    if (values.map(String).includes(String(selected))) select.value = String(selected);
  }

  function selectedModel() { return models.find(model => model.id === $('model').value); }

  function sizeLabel(size) {
    const [w, h] = size.split('x').map(Number);
    return `${w} × ${h}`;
  }

  const qwenEdit = 'qwen-image-2.1-turbo-edit';
  const references = [];
  let nextReferenceId = 1;
  let draggedReferenceIndex = null;
  let inputSize = null;
  const referenceLimit = (model = selectedModel()) => model?.reference_limit || 1;
  const referenceBudget = (model = selectedModel()) => model?.reference_token_limit || 0;
  const usesNativeReferences = (model = selectedModel()) => model?.native_references === true;
  const isMultiReference = () => referenceLimit() > 1;
  const generationDimensions = {};
  const editDimensions = {};
  let previousSize;

  function toggleMatchSize() {
    const match = $('source-size').checked;
    for (const id of ['edit-preset', 'lock-ratio']) $(id).disabled = match;
    for (const id of ['width', 'height']) $(id).readOnly = match;
    if (match) {
      previousSize = [$('width').value, $('height').value];
    } else if (previousSize) {
      [$('width').value, $('height').value] = previousSize;
      previousSize = null;
    }
    updateSendingSize();
  }

  function applyEditPreset() {
    const value = $('edit-preset').value;
    if (value === 'custom') return;
    let sides = value.split('x').map(Number);
    if ($('lock-ratio').checked) {
      const scale = Math.min(sides[0] / inputSize.width, sides[1] / inputSize.height);
      sides = scaledInput(scale, selectedModel().size_limits);
    }
    [$('width').value, $('height').value] = sides;
  }

  function scaledInput(scale, limits) {
    const first = Math.ceil(limits.min / limits.multiple) * limits.multiple;
    const last = Math.floor(limits.max / limits.multiple) * limits.multiple;
    const snap = side => Math.min(last, Math.max(first, Math.round(side * scale / limits.multiple) * limits.multiple));
    return [snap(inputSize.width), snap(inputSize.height)];
  }

  function applyInputSize(edge) {
    const { size_limits: limits } = selectedModel();
    const longest = Math.max(inputSize.width, inputSize.height);
    const max = Math.floor(limits.max / limits.multiple) * limits.multiple;
    const target = edge === Infinity ? Math.min(longest, max) : Math.min(edge, max);
    const [width, height] = scaledInput(target / longest, limits);
    $('width').value = width;
    $('height').value = height;
  }

  function describeInput(width, height) {
    let a = width, b = height;
    while (b) [a, b] = [b, a % b];
    return `${width} × ${height} px · ${width / a}:${height / a}`;
  }

  function syncCustomRatio(anchor = 'width', commit = false) {
    if (!$('lock-ratio').checked || $('edit-size').hidden || $('source-size').checked) return;
    const { size_limits: limits } = selectedModel();
    const value = Number($(anchor).value);
    if (value <= 0) return;
    const max = Math.floor(limits.max / limits.multiple) * limits.multiple;
    const scale = Math.min(value / inputSize[anchor], max / Math.max(inputSize.width, inputSize.height));
    const [width, height] = scaledInput(scale, limits);
    if (commit || anchor !== 'width') $('width').value = width;
    if (commit || anchor !== 'height') $('height').value = height;
  }

  function fillSizes(model) {
    const previous = $('size').value;
    const options = model.supported_sizes.map(size => new Option(sizeLabel(size), size));
    const limits = model.size_limits;
    if (limits) {
      options.push(new Option('Custom…', 'custom'));
      for (const id of ['width', 'height']) {
        Object.assign($(id), { min: limits.min, max: limits.max, step: limits.multiple });
      }
    }
    $('size').replaceChildren(...options);
    if (options.some(option => option.value === previous)) $('size').value = previous;
    $('size-help').textContent = limits
      ? `${limits.min}–${limits.max} px · step ${limits.multiple}`
      : 'Sizes supported by this model.';
    updateCustomSize();
  }

  function updateCustomSize() {
    const limits = selectedModel()?.size_limits;
    const edit = Boolean(selectedModel()?.edits_image && limits);
    const ready = edit && Boolean(inputSize);
    const dimensions = $('custom-size');
    if (edit !== (dimensions.parentElement.id === 'edit-dimensions')) {
      const nextValues = edit ? editDimensions : generationDimensions;
      const previousValues = edit ? generationDimensions : editDimensions;
      for (const side of ['width', 'height']) {
        previousValues[side] = $(side).value;
        $(side).value = nextValues[side];
      }
      (edit ? $('edit-dimensions') : $('generation-dimensions')).append(dimensions);
      if (ready && !editDimensions.width) applyInputSize(1024);
    }
    $('edit-size').hidden = !ready;
    $('size').disabled = edit;
    $('size').closest('.field').hidden = $('generation-size').hidden = edit;
    if (ready) {
      $('edit-size-help').textContent = `${limits.min}–${limits.max} px · step ${limits.multiple}`;
      const preset = $('edit-preset');
      fillOptions(preset, ['custom', ...selectedModel().supported_sizes], preset.value);
      preset.options[0].text = 'Custom…';
      for (const option of [...preset.options].slice(1)) option.text = sizeLabel(option.value);
      updateSendingSize();
    }
    for (const side of ['width', 'height']) $(side).readOnly = edit && $('source-size').checked;
    const custom = edit ? ready : $('size').value === 'custom';
    $('custom-size').hidden = !custom;
    $('width').disabled = $('height').disabled = !custom;
  }

  function chosenSize(model) {
    if ($('size').value !== 'custom' && !(model.edits_image && model.size_limits)) return $('size').value;
    const limits = model.size_limits;
    const sides = [Number($('width').value), Number($('height').value)];
    if (sides.some(side => !Number.isInteger(side) || side < limits.min || side > limits.max || side % limits.multiple)) {
      throw new Error(`Each side must be ${limits.min}–${limits.max} px · step ${limits.multiple}.`);
    }
    return `${sides[0]}x${sides[1]}`;
  }

  function fillLoras(model) {
    $('lora-field').hidden = !model.loras && !model.lora_error;
    $('lora-error').hidden = !model.lora_error;
    $('lora-error').textContent = model.lora_error ? `Cannot load styles. ${model.lora_error}` : '';
    const files = { ok: 'Ready', download: 'Downloads on first use', missing: 'Unavailable' };
    $('lora-list').replaceChildren(...(model.loras || []).map(lora => {
      const item = document.createElement('div');
      const row = document.createElement('div');
      row.className = 'lora-row';
      const info = document.createElement('div');
      info.className = 'lora-info';
      const label = document.createElement('label');
      const box = document.createElement('input');
      const name = document.createElement('span');
      const availability = document.createElement('span');
      item.className = 'lora';
      item.dataset.missing = String(lora.status === 'missing');
      box.type = 'checkbox';
      box.value = lora.name;
      box.checked = lora.enabled && lora.status !== 'missing';
      box.disabled = lora.status === 'missing';
      name.className = 'lora-name';
      name.textContent = lora.name;
      availability.className = 'badge';
      availability.textContent = files[lora.status] || lora.status;
      label.title = availability.textContent;
      box.setAttribute('aria-label', `${lora.name}: ${availability.textContent}`);
      label.append(box, name);
      row.append(label);
      item.append(row);
      info.append(availability);
      if (lora.trigger) {
        const trigger = document.createElement('p');
        trigger.className = 'lora-trigger';
        trigger.textContent = `Prompt words: ${lora.trigger}`;
        const add = document.createElement('button');
        add.type = 'button';
        add.className = 'lora-add';
        add.textContent = 'Add words';
        add.setAttribute('aria-label', `Add ${lora.name} to prompt`);
        add.disabled = box.disabled;
        add.addEventListener('click', () => {
          const words = lora.trigger.split(',').map(word => word.trim()).filter(Boolean);
          const existing = $('prompt').value.toLocaleLowerCase();
          const missing = words.filter(word => !existing.includes(word.toLocaleLowerCase()));
          if (missing.length) {
            const prompt = $('prompt');
            const text = prompt.value;
            prompt.setRangeText((text.trim() ? ', ' : '') + missing.join(', '), text.length, text.length);
          }
          box.checked = true;
          $('prompt').focus();
          status(`${lora.name} selected.`);
        });
        info.append(trigger);
        row.append(add);
      }
      const details = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = 'Details';
      summary.setAttribute('aria-label', `${lora.name} details`);
      details.append(summary, info);
      for (const line of [`File: ${lora.file}`, `Strength: ${Number(lora.scale).toFixed(2)}`, `License: ${lora.license || 'Not specified'}`]) {
        const small = document.createElement('small');
        small.textContent = line;
        info.append(small);
      }
      item.append(details);
      return item;
    }));
    if (!model.loras?.length && model.loras) {
      const empty = document.createElement('small');
      empty.className = 'help';
      empty.textContent = 'No styles are available for this model.';
      $('lora-list').append(empty);
    }
  }

  function updateModel(preserve = false) {
    const model = selectedModel();
    if (!model) return;
    const styles = preserve
      ? new Set([...$('lora-list').querySelectorAll('input:checked')].map(box => box.value))
      : new Set();
    const previousSteps = Number($('steps').value);
    const steps = preserve && model.supported_steps.includes(previousSteps) ? previousSteps : model.default_steps;
    fillOptions($('steps'), model.supported_steps, steps);
    $('steps').disabled = model.supported_steps.length < 2;
    $('steps-help').hidden = model.supported_steps.length > 1;
    fillSizes(model);
    fillLoras(model);
    if (preserve) {
      for (const box of $('lora-list').querySelectorAll('input')) {
        box.checked = !box.disabled && styles.has(box.value);
      }
    }
    const count = model.loras?.length || 0;
    $('style-count').textContent = count ? `(${count})` : '';
    if (!preserve) $('seed').value = model.default_seed;
    $('input-field').hidden = !model.edits_image;
    $('input-image').disabled = !model.edits_image;
    $('input-image').multiple = isMultiReference();
    $('input-image').accept = usesNativeReferences(model) ? 'image/png,image/jpeg' : 'image/png,image/jpeg,image/bmp,image/gif';
    renderReferences();
    updateAction(model);
  }

  async function loadModels() {
    const previous = $('model').value;
    disableControls(true);
    $('retry').hidden = true;
    setConnection('connecting', 'Connecting');
    status('Connecting to your server…');
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    try {
      const data = await readResponse(await fetch(new URL('models', api), { signal: controller.signal, cache: 'no-store' }));
      if (!Array.isArray(data.data) || !data.data.length) throw new Error('No models available.');
      models = data.data;
      const groups = [];
      for (const [edits, label] of [[false, 'Generation'], [true, 'Editing']]) {
        const group = document.createElement('optgroup');
        group.label = label;
        for (const model of models.filter(model => Boolean(model.edits_image) === edits)) {
          group.append(new Option(model.id, model.id));
        }
        if (group.children.length) groups.push(group);
      }
      $('model').replaceChildren(...groups);
      const preferred = models.find(model => model.id === previous) || models.find(model => model.is_default) || models[0];
      $('model').value = preferred.id;
      updateModel(preferred.id === previous);
      disableControls(false);
      setConnection('connected', 'Connected');
      status('Ready');
    } catch (error) {
      setConnection('error', 'Disconnected');
      status(`Cannot load models. ${error.message} `, true);
      $('retry').hidden = false;
    } finally { clearTimeout(timeout); }
  }

  $('random-seed').addEventListener('click', () => {
    const random = crypto.getRandomValues(new Uint32Array(1))[0] & 0x7fffffff;
    const previous = Number($('seed').value);
    $('seed').value = random === previous ? (random + 1) % 2147483648 : random;
    status('New seed selected.');
  });
  $('generation-form').addEventListener('keydown', event => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey) && !event.isComposing) {
      event.preventDefault();
      if ($('controls').disabled) return;
      if (event.target.id === 'width' || event.target.id === 'height') syncCustomRatio(event.target.id, true);
      $('generation-form').requestSubmit();
    }
  });
  $('generation-form').addEventListener('invalid', event => {
    let section = event.target.closest('details');
    while (section) { section.open = true; section = section.parentElement.closest('details'); }
    status(event.target.validationMessage, true);
  }, true);
  $('model').addEventListener('change', () => updateModel());
  $('size').addEventListener('change', updateCustomSize);
  $('edit-preset').addEventListener('change', applyEditPreset);
  $('source-size').addEventListener('change', toggleMatchSize);
  $('lock-ratio').addEventListener('change', () => {
    if ($('edit-preset').value === 'custom') syncCustomRatio('width', true);
    else applyEditPreset();
  });
  for (const side of ['width', 'height']) {
    $(side).addEventListener('input', () => {
      if (!$('edit-size').hidden) $('edit-preset').value = 'custom';
      syncCustomRatio(side);
    });
    $(side).addEventListener('change', () => syncCustomRatio(side, true));
  }
  $('refresh').addEventListener('click', loadModels);
  $('retry').addEventListener('click', loadModels);
  $('input-image').addEventListener('change', event => {
    const field = event.target;
    const files = [...field.files];
    field.value = '';
    if (files.length) addReferences(files);
  });

  function showDetails(metadata, elapsed) {
    const labels = {
      '0_Model': 'Model', '1_Image_H': 'Height', '2_Image_W': 'Width',
      '3_Denoising_steps': 'Steps', '4_Seed': 'Seed', '5_Text_encoder': 'Text encoder',
      '6_Denoising': 'Denoising', '7_Vae_decoder': 'VAE decoder',
      '8_E2E_runtime': 'E2E runtime', '9_Worker_wall_time': 'Worker time (including loading)',
      '10_LoRAs': 'LoRAs'
    };
    $('details').replaceChildren();
    for (const [key, label] of Object.entries(labels)) {
      if (metadata[key] === undefined) continue;
      const group = document.createElement('div');
      const term = document.createElement('dt');
      const value = document.createElement('dd');
      term.textContent = label;
      value.textContent = String(metadata[key]).replace(/(\d)s\b/g, '$1 s');
      group.append(term, value);
      $('details').append(group);
    }
    const time = document.createElement('div');
    const term = document.createElement('dt');
    const value = document.createElement('dd');
    term.textContent = 'Request time';
    value.textContent = `${elapsed.toFixed(2)} s`;
    time.append(term, value);
    $('details').append(time);
  }

  $('generation-form').addEventListener('submit', async event => {
    event.preventDefault();
    if ($('controls').disabled) return;
    const model = selectedModel();
    const inputs = activeReferences();
    if (model.edits_image && !inputs.length) {
      status('Add a reference image.', true);
      return;
    }
    const limit = referenceLimit(model);
    if (model.edits_image && inputs.length > limit) {
      status('Remove references to fit the ' + limit + '-image limit.', true);
      return;
    }
    const plan = model.edits_image ? referencePlan() : [];
    const budget = referenceBudget(model);
    if (budget && tokenCount(plan) > budget) {
      status('Choose Auto or a smaller input limit to fit the reference token budget.', true);
      return;
    }
    const prompt = promptForModel(model).trim();
    if (model.edits_image && $('prompt').querySelector('[data-missing]')) {
      status('Add the missing reference or remove its mention.', true);
      return;
    }
    if (!prompt || new TextEncoder().encode(prompt).length > 16384 || prompt.includes('\0')) {
      status('Prompt must be 1–16384 bytes, without NUL.', true);
      return;
    }
    let size;
    try { size = chosenSize(model); }
    catch (error) { status(error.message, true); return; }
    const params = {
      model: model.id, prompt, size, n: 1,
      response_format: $('response-format').value,
      steps: Number($('steps').value), seed: Number($('seed').value)
    };
    if (model.loras) params.loras = [...$('lora-list').querySelectorAll('input:checked')].map(box => box.value);
    let body;
    let editInput = null;
    const headers = { Accept: 'application/json' };
    if (model.edits_image) {
      body = new FormData();
      for (const [key, value] of Object.entries(params)) body.append(key, String(value));

    } else {
      headers['Content-Type'] = 'application/json';
      body = JSON.stringify(params);
    }
    disableControls(true);
    $('generation-form').setAttribute('aria-busy', 'true');
    const started = performance.now();
    const elapsed = () => (performance.now() - started) / 1000;
    $('preview-panel').dataset.busy = 'true';
    $('preview-panel').setAttribute('aria-busy', 'true');
    $('generate').textContent = model.edits_image ? 'Editing…' : 'Generating…';
    if (!outputUrl) {
      $('preview-message').textContent = model.edits_image ? 'Applying your changes…' : 'Creating your image…';
      $('preview-hint').textContent = 'Loading the model may take a moment.';
    }
    status(`${model.edits_image ? 'Editing' : 'Generating'} your image…`);
    const progress = () => { $('preview-state').textContent = `${model.edits_image ? 'Editing' : 'Generating'} · ${elapsed().toFixed(0)} s`; };
    progress();
    const timer = setInterval(progress, 1000);
    let nextUrl = '';
    let nextBeforeUrl = '';
    try {
      if (model.edits_image) {
        for (const dimensions of plan) {
          const image = await scaleReference(dimensions);
          editInput ||= image;
          body.append('image', image);
        }
      }
      const route = model.edits_image ? 'images/edits' : 'images/generations';
      const data = await readResponse(await fetch(new URL(route, api), { method: 'POST', headers, body }));
      const item = data.data?.[0];
      let blob;
      if (item?.b64_json) {
        const bytes = Uint8Array.from(atob(item.b64_json), char => char.charCodeAt(0));
        blob = new Blob([bytes], { type: 'image/png' });
      } else if (item?.url) {
        const url = new URL(item.url, location.origin);
        if (!url.pathname.startsWith('/images/')) throw new Error('Unexpected image URL.');
        const response = await fetch(url.pathname + url.search);
        if (!response.ok) throw new Error(`Cannot load image (HTTP ${response.status}).`);
        blob = await response.blob();
      } else { throw new Error('No image returned.'); }
      nextUrl = URL.createObjectURL(blob);
      const decoded = new Image();
      decoded.src = nextUrl;
      await decoded.decode();
      nextBeforeUrl = await comparisonInput(editInput);
      if (outputUrl) URL.revokeObjectURL(outputUrl);
      if (beforeUrl) URL.revokeObjectURL(beforeUrl);
      outputUrl = nextUrl;
      nextUrl = '';
      beforeUrl = nextBeforeUrl;
      nextBeforeUrl = '';
      $('output').src = outputUrl;
      $('output').alt = `${model.edits_image ? 'Edited' : 'Generated'} image: ${prompt}`;
      $('image-frame').style.setProperty('--image-ratio', decoded.naturalWidth / decoded.naturalHeight);
      if (beforeUrl) $('before-image').src = beforeUrl;
      else $('before-image').removeAttribute('src');
      $('comparison-tools').hidden = !beforeUrl;
      $('comparison-unavailable').hidden = !editInput || Boolean(beforeUrl);
      updateComparisonPosition(50);
      showComparison(Boolean(beforeUrl));
      $('result-heading').textContent = model.edits_image ? 'Edited image' : 'Generated image';
      $('download').href = outputUrl;
      $('open-image').href = outputUrl;
      $('download').download = `vigenflow-${model.id}-${params.seed}.png`;
      showDetails(data.metadata || {}, elapsed());
      const e2e = data.metadata?.['8_E2E_runtime'];
      const runtime = e2e != null ? String(e2e).replace(/(\d)s\b/g, '$1 s') : `${elapsed().toFixed(2)} s`;
      const runtimeLabel = e2e != null ? 'E2E runtime' : 'Request time';
      $('result-summary').textContent = `${decoded.naturalWidth} × ${decoded.naturalHeight} px · Seed ${params.seed} · ${runtimeLabel}: ${runtime}`;
      $('output-prompt').textContent = prompt;
      $('preview-state').textContent = 'Image ready';
      $('result').hidden = false;
      $('preview-placeholder').hidden = true;
      clearInterval(timer);
      status(`Done in ${runtime} (${e2e != null ? 'E2E' : 'request time'}). Your image is ready.`);
      if (matchMedia('(max-width: 760px)').matches) {
        $('preview-panel').scrollIntoView({ block: 'start', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
      }
    } catch (error) {
      clearInterval(timer);
      if (nextUrl) URL.revokeObjectURL(nextUrl);
      if (nextBeforeUrl) URL.revokeObjectURL(nextBeforeUrl);
      $('preview-state').textContent = outputUrl ? 'Previous image' : 'Try again';
      if (!outputUrl) {
        $('preview-message').textContent = 'Your image could not be created';
        $('preview-hint').textContent = 'Check the error below.';
      }
      status(`Image request failed: ${error.message}`, true);
    } finally {
      clearInterval(timer);
      disableControls(false);
      $('generation-form').removeAttribute('aria-busy');
      $('preview-panel').dataset.busy = 'false';
      $('preview-panel').removeAttribute('aria-busy');
      $('generate').textContent = model.edits_image ? 'Edit image' : 'Generate image';
    }
  });
  const composer = $('prompt');
  const nameSegments = new Intl.Segmenter(undefined, { granularity: 'grapheme' });
  let lastSelection = [0, 0];

  function editorOffsets() {
    const range = getSelection().rangeCount && getSelection().getRangeAt(0);
    if (!range || !composer.contains(range.commonAncestorContainer)) return lastSelection;
    const prefix = document.createRange();
    prefix.selectNodeContents(composer);
    prefix.setEnd(range.startContainer, range.startOffset);
    const start = prefix.toString().length;
    return lastSelection = [start, start + range.toString().length];
  }

  function editorPoint(offset) {
    for (const node of composer.childNodes) {
      const length = node.textContent.length;
      if (offset <= length) {
        if (node.nodeType === Node.TEXT_NODE) return [node, offset];
        return [composer, Array.prototype.indexOf.call(composer.childNodes, node) + Number(offset > 0)];
      }
      offset -= length;
    }
    return [composer, composer.childNodes.length];
  }

  function editorRange(start, end) {
    const range = document.createRange();
    range.setStart(...editorPoint(start));
    range.setEnd(...editorPoint(end));
    return range;
  }

  function selectEditor(start, end = start) {
    const range = editorRange(start, end);
    getSelection().removeAllRanges();
    getSelection().addRange(range);
    lastSelection = [start, end];
  }

  function replaceEditor(value, start, end) {
    const range = editorRange(start, end);
    range.deleteContents();
    const node = typeof value === 'string' ? document.createTextNode(value) : value;
    const last = node.nodeType === Node.DOCUMENT_FRAGMENT_NODE ? node.lastChild : node;
    range.insertNode(node);
    composer.focus();
    range.setStartAfter(last);
    range.collapse(true);
    getSelection().removeAllRanges();
    getSelection().addRange(range);
    editorOffsets();
  }

  Object.defineProperties(composer, {
    value: {
      get: () => composer.textContent,
      set(value) {
        composer.textContent = value;
        lastSelection = [value.length, value.length];
      }
    },
    selectionStart: { get: () => editorOffsets()[0] },
    selectionEnd: { get: () => editorOffsets()[1] },
    disabled: {
      get: () => !composer.isContentEditable,
      set: disabled => { composer.contentEditable = disabled ? 'false' : 'plaintext-only'; }
    },
    placeholder: { set: value => { composer.dataset.placeholder = value; } }
  });
  composer.setSelectionRange = selectEditor;
  composer.setRangeText = replaceEditor;
  composer.addEventListener('blur', editorOffsets);

  function activeReferences() {
    return isMultiReference() ? references : references.slice(0, 1);
  }

  function referenceChip(position, reference = activeReferences()[position - 1]) {
    const chip = document.createElement('span');
    chip.contentEditable = 'false';
    chip.className = 'ref-chip';
    chip.dataset.position = position;
    if (reference) chip.dataset.reference = reference.id;
    const thumbnail = new Image();
    thumbnail.alt = '';
    chip.append(thumbnail, document.createTextNode(''));
    refreshReferenceChip(chip);
    return chip;
  }

  function presetContent(content) {
    if (!selectedModel()?.edits_image) return content;
    const parts = content.split(/(<image[1-9]\d*>|\bImage [1-9]\d*\b)/gi);
    if (parts.length === 1) return content;
    const fragment = document.createDocumentFragment();
    parts.forEach((part, index) => {
      if (index % 2) fragment.append(referenceChip(Number(part.replace(/\D/g, ''))));
      else if (part) fragment.append(part);
    });
    return fragment;
  }

  function promptForModel(model) {
    if (!model.edits_image) return composer.value;
    const copy = composer.cloneNode(true);
    copy.querySelectorAll('.ref-chip').forEach(chip => {
      const position = chip.dataset.position;
      const reference = model.id === qwenEdit ? '<image' + position + '>' : 'image ' + position;
      chip.replaceWith(reference);
    });
    return copy.textContent;
  }

  function insertReference(range, reference, prefix = '') {
    const position = activeReferences().indexOf(reference) + 1;
    if (composer.value !== range.value || !position) return;
    const chip = referenceChip(position, reference);
    replaceEditor(chip, range.start, range.end);
    if (prefix) chip.before(prefix);
    const next = range.value[range.end];
    const suffix = !next || /[\p{L}\p{N}]/u.test(next) ? ' ' : '';
    if (suffix) chip.after(suffix);
    selectEditor(range.start + prefix.length + chip.textContent.length + suffix.length);
  }

  function refreshReferenceChip(chip) {
    const inputs = activeReferences();
    const reference = chip.dataset.reference
      ? inputs.find(item => item.id === Number(chip.dataset.reference))
      : inputs[Number(chip.dataset.position) - 1];
    if (reference) {
      chip.dataset.reference = reference.id;
      chip.dataset.position = inputs.indexOf(reference) + 1;
    }
    const tag = 'Image ' + chip.dataset.position;
    chip.lastChild.textContent = tag;
    const thumbnail = chip.querySelector('img');
    if (reference && thumbnail.getAttribute('src') !== reference.url) thumbnail.src = reference.url;
    if (!reference) thumbnail.removeAttribute('src');
    chip.toggleAttribute('data-missing', !reference);
    const name = reference?.name.trim() || tag;
    const letters = Array.from(nameSegments.segment(name), entry => entry.segment);
    const shortName = letters.length > 30 ? letters.slice(0, 29).join('') + '…' : name;
    chip.dataset.label = reference ? '@' + shortName : tag;
    chip.title = reference ? '@' + name : 'Add ' + tag;
    chip.setAttribute('aria-label', chip.title);
  }

  function refreshReferenceChips() {
    composer.querySelectorAll('.ref-chip').forEach(refreshReferenceChip);
  }

  function referenceButton(label, action, description = label) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.setAttribute('aria-label', description);
    if (action) button.addEventListener('click', action);
    return button;
  }

  function renderReferences() {
    const inputs = activeReferences();
    inputSize = inputs[0] || null;
    const list = $('reference-list');
    list.replaceChildren();
    $('ref-upload-label').textContent = isMultiReference() ? 'Add reference' : inputSize ? 'Replace' : 'Add image';
    const limit = referenceLimit();
    $('input-help').hidden = Boolean(inputSize) && inputs.length <= limit;
    const formats = usesNativeReferences() ? 'PNG or JPEG' : 'PNG, JPEG, BMP or GIF';
    const uploadHint = limit > 1 ? 'Drop or paste up to ' + limit + ' references.' : 'Drop or paste an image.';
    $('input-help').textContent = inputs.length > limit
      ? 'Remove references to fit the ' + limit + '-image limit.'
      : uploadHint + ' ' + formats + ', up to 40 MB each.';
    $('reference-size').hidden = !inputSize;
    $('reference-budget-help').hidden = !referenceBudget();
    if (!inputSize) {
      $('source-size').checked = false;
      toggleMatchSize();
    }
    for (const [index, reference] of inputs.entries()) {
      const tag = 'Image ' + (index + 1);
      const card = document.createElement('div');
      card.className = 'ref-card';
      const preview = referenceButton('', () => openImageView({
        url: reference.url, name: reference.file.name,
        alt: 'Original reference', downloadLabel: 'Download original'
      }), 'Open ' + tag);
      preview.className = 'open-reference';
      preview.setAttribute('aria-haspopup', 'dialog');
      const thumbnail = new Image();
      thumbnail.className = 'input-preview';
      thumbnail.alt = '';
      thumbnail.src = reference.url;
      preview.append(thumbnail);
      const info = document.createElement('div');
      const heading = document.createElement('div');
      heading.className = 'ref-heading';
      const label = document.createElement('label');
      label.textContent = tag;
      const name = document.createElement('input');
      name.id = 'reference-name-' + reference.id;
      label.htmlFor = name.id;
      name.className = 'ref-name';
      name.maxLength = 255;
      name.value = reference.name;
      name.title = reference.name;
      name.addEventListener('input', () => {
        reference.name = name.value;
        name.title = name.value;
        refreshReferenceChips();
      });
      name.addEventListener('keydown', event => {
        if (event.key === 'Enter') event.preventDefault();
      });
      heading.append(label);
      if (isMultiReference() && inputs.length > 1) {
        addReferenceDragHandle(card, heading, index, tag);
      }
      const original = document.createElement('small');
      original.className = 'help';
      original.textContent = describeInput(reference.width, reference.height);
      const sending = document.createElement('small');
      sending.className = 'help reference-sending';
      sending.hidden = !isMultiReference();
      const insert = referenceButton('@', () => {
        const { value, selectionStart: start, selectionEnd: end } = composer;
        hidePicker();
        insertReference({ value, start, end }, reference, start && !/\s/.test(value[start - 1]) ? ' ' : '');
      }, 'Insert ' + tag);
      const remove = referenceButton('×', () => removeReference(index), 'Remove ' + tag);
      remove.title = 'Remove ' + tag;
      const fields = document.createElement('div');
      fields.className = 'ref-fields';
      fields.append(name, insert, remove);
      info.append(heading, fields, original, sending);
      card.append(preview, info);
      list.append(card);
    }
    refreshReferenceChips();
    if (inputSize) updateInputLimits();
    if (selectedModel()) updateCustomSize();
  }

  function moveReference(index, next) {
    if (next === index || next < 0 || next >= references.length) return;
    const [reference] = references.splice(index, 1);
    references.splice(next, 0, reference);
    renderReferences();
  }

  function clearReferenceDrag() {
    draggedReferenceIndex = null;
    const list = $('reference-list');
    list.querySelector('[data-dragging]')?.removeAttribute('data-dragging');
    list.querySelector('[data-drop-position]')?.removeAttribute('data-drop-position');
  }

  function addReferenceDragHandle(card, heading, index, tag) {
    const handle = referenceButton('⠿', null, 'Reorder ' + tag);
    handle.className = 'ref-drag';
    handle.draggable = true;
    handle.title = 'Drag to reorder, or use the Up and Down arrow keys.';
    handle.addEventListener('keydown', event => {
      if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
      event.preventDefault();
      const next = index + (event.key === 'ArrowUp' ? -1 : 1);
      moveReference(index, next);
      $('reference-list').children[next]?.querySelector('.ref-drag')?.focus();
    });
    handle.addEventListener('dragstart', event => {
      draggedReferenceIndex = index;
      event.dataTransfer.setData('application/x-vigenflow-reference', String(index));
      event.dataTransfer.effectAllowed = 'move';
      card.dataset.dragging = '';
    });
    handle.addEventListener('dragend', clearReferenceDrag);
    card.addEventListener('dragover', event => {
      if (draggedReferenceIndex === null || draggedReferenceIndex === index) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = 'move';
      const bounds = card.getBoundingClientRect();
      $('reference-list').querySelector('[data-drop-position]')?.removeAttribute('data-drop-position');
      card.dataset.dropPosition = event.clientY < bounds.top + bounds.height / 2 ? 'before' : 'after';
    });
    card.addEventListener('dragleave', event => {
      if (!card.contains(event.relatedTarget)) delete card.dataset.dropPosition;
    });
    card.addEventListener('drop', event => {
      const source = draggedReferenceIndex;
      if (source === null || !card.dataset.dropPosition) return;
      event.preventDefault();
      const insertion = index + Number(card.dataset.dropPosition === 'after');
      clearReferenceDrag();
      moveReference(source, insertion - Number(source < insertion));
    });
    heading.append(handle);
  }

  function removeReference(index) {
    const [removed] = references.splice(index, 1);
    URL.revokeObjectURL(removed.url);
    if (!references.length) {
      composer.querySelectorAll('.ref-chip').forEach(chip => {
        if (Number(chip.dataset.reference) === removed.id) delete chip.dataset.reference;
      });
    }
    renderReferences();
  }

  async function addReferences(files) {
    if (!selectedModel()?.edits_image || $('controls').disabled) return;
    const limit = referenceLimit();
    const multi = limit > 1;
    if (multi ? references.length + files.length > limit : files.length !== 1) {
      status(multi ? 'Add up to ' + limit + ' references.' : 'Add one image at a time.', true);
      return;
    }
    disableControls(true);
    const loaded = [];
    const hadInput = Boolean(inputSize);
    const pngOrJpeg = usesNativeReferences();
    try {
      for (const file of files) {
        const pattern = pngOrJpeg ? /\.(png|jpe?g)$/i : /\.(png|jpe?g|bmp|gif)$/i;
        const types = pngOrJpeg ? /^image\/(png|jpeg)$/ : /^image\/(png|jpeg|bmp|x-ms-bmp|gif)$/;
        if (!pattern.test(file.name) && !types.test(file.type)) {
          throw new Error(pngOrJpeg ? 'Choose a PNG or JPEG image.' : 'Choose a PNG, JPEG, BMP or GIF image.');
        }
        if (file.size > 40 * 1024 * 1024) throw new Error(file.name + ' must be at most 40 MB.');
        const url = URL.createObjectURL(file);
        const name = file.name.replace(/\.[^.]+$/, '') || file.name;
        const reference = { file, url, name, id: nextReferenceId++ };
        loaded.push(reference);
        const image = new Image();
        image.src = url;
        await image.decode();
        reference.image = image;
        reference.width = image.naturalWidth;
        reference.height = image.naturalHeight;
        if (reference.width * reference.height > 40000000) throw new Error(file.name + ' must be at most 40 megapixels.');
      }
      if (!multi && references.length) {
        loaded[0].id = references[0].id;
        URL.revokeObjectURL(references[0].url);
        references[0] = loaded[0];
      } else {
        references.push(...loaded);
      }
      renderReferences();
      if ((!hadInput || !multi) && !$('source-size').checked && !$('edit-size').hidden) applyInputSize(1024);
      status('Ready');
    } catch (error) {
      for (const reference of loaded) URL.revokeObjectURL(reference.url);
      status(error instanceof DOMException ? 'Could not read the reference image.' : error.message, true);
    } finally {
      disableControls(false);
    }
  }

  const inputLimit = $('input-limit');
  const limitOption = value => inputLimit.querySelector('[value="' + value + '"]');
  let preferredInputLimit;

  function preparedDimensions(reference, factor, align) {
    const width = Math.max(1, Math.round(reference.width * factor));
    const height = Math.max(1, Math.round(reference.height * factor));
    const trim = side => align ? Math.max(16, Math.floor(side / 16) * 16) : side;
    return { reference, width, height, targetWidth: trim(width), targetHeight: trim(height) };
  }

  function referenceTokens({ targetWidth, targetHeight }) {
    return targetWidth / 16 * (targetHeight / 16);
  }

  function tokenCount(plan) {
    return plan.reduce((sum, image) => sum + referenceTokens(image), 0);
  }

  function referencePlan(limit = inputLimit.querySelector(':checked').value) {
    const inputs = activeReferences();
    const budget = referenceBudget();
    const align = usesNativeReferences() || Boolean(budget) || $('source-size').checked;
    const longEdge = limit === 'max' ? 2048 : Number(limit);
    const full = inputs.map(reference => preparedDimensions(
      reference, Math.min(1, longEdge / Math.max(reference.width, reference.height)), align
    ));
    if (limit !== 'max' || !budget || tokenCount(full) <= budget) return full;
    const planAt = areaLimit => full.map(image => {
      const factor = Math.min(1, Math.sqrt(areaLimit / (image.width * image.height)));
      const dimensions = preparedDimensions({ width: image.width, height: image.height }, factor, align);
      return { ...dimensions, reference: image.reference };
    });
    let lower = 0;
    let upper = Math.max(...full.map(image => image.width * image.height));
    for (let i = 0; i < 24; i++) {
      const midpoint = (lower + upper) / 2;
      if (tokenCount(planAt(midpoint)) <= budget) lower = midpoint;
      else upper = midpoint;
    }
    return planAt(lower);
  }

  function updateInputLimits() {
    (limitOption(preferredInputLimit) || limitOption('max')).checked = true;
    updateSendingSize();
  }

  function updateSendingSize() {
    if (!inputSize) return;
    const plan = referencePlan();
    const first = plan[0];
    if (!first) return;
    const budget = referenceBudget();
    const label = dimensions => dimensions.targetWidth + ' × ' + dimensions.targetHeight + ' px';
    $('sending-size').textContent = plan.length === 1 ? 'Sending: ' + label(first) : '';
    $('reference-list').querySelectorAll('.reference-sending').forEach((element, index) => {
      if (isMultiReference() && plan[index]) {
        const tokens = budget ? ' · ' + referenceTokens(plan[index]).toLocaleString() + ' tokens' : '';
        element.textContent = 'Sending: ' + label(plan[index]) + tokens;
      }
    });
    if ($('source-size').checked && selectedModel()?.edits_image) {
      $('width').value = first.targetWidth;
      $('height').value = first.targetHeight;
    }
    const budgetInfo = $('reference-budget');
    budgetInfo.hidden = !budget;
    if (!budget) return;
    const tokens = tokenCount(plan);
    budgetInfo.toggleAttribute('data-over-limit', tokens > budget);
    budgetInfo.textContent = tokens.toLocaleString() + ' / ' + budget.toLocaleString() + ' reference tokens' +
      (tokens > budget ? ' · Over budget. Choose Auto or a smaller limit.' : '');
  }

  inputLimit.addEventListener('change', event => {
    preferredInputLimit = event.target.value;
    updateSendingSize();
  });

  async function scaleReference({ reference, width, height, targetWidth, targetHeight }) {
    const native = usesNativeReferences();
    const match = $('source-size').checked;
    if (width === reference.width && height === reference.height &&
        targetWidth === width && targetHeight === height &&
        (native ? /^image\/(png|jpeg)$/.test(reference.file.type) : !match)) {
      return reference.file;
    }
    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(reference.image, Math.ceil((targetWidth - width) / 2), Math.ceil((targetHeight - height) / 2), width, height);
    const jpeg = (native || !match) && reference.file.type === 'image/jpeg';
    const blob = await new Promise(resolve => canvas.toBlob(resolve, jpeg ? 'image/jpeg' : 'image/png', 0.94));
    if (!blob) throw new Error('Could not resize ' + reference.file.name + '.');
    return new File([blob], reference.file.name.replace(/\.[^.]+$/, '') + (jpeg ? '.jpg' : '.png'), { type: blob.type });
  }
  const sendingInfo = $('sending-info');
  const sendingDetails = sendingInfo.parentElement;
  sendingInfo.addEventListener('keydown', event => {
    if (event.key === 'Escape') sendingDetails.dataset.dismissed = '';
  });
  sendingInfo.addEventListener('focus', () => delete sendingDetails.dataset.dismissed);
  sendingInfo.addEventListener('pointerenter', () => delete sendingDetails.dataset.dismissed);
  const referenceView = $('reference-view');

  function openImageView({ url, name, label, details, alt, downloadLabel, prompt = '' }) {
    referenceView.setAttribute('aria-label', label || 'Input image preview');
    $('reference-info').hidden = !label;
    $('reference-title').textContent = label || '';
    $('reference-original').textContent = details || '';
    $('viewer-prompt').hidden = !prompt;
    $('viewer-used-prompt').textContent = prompt;
    const image = $('full-reference');
    image.src = url;
    image.alt = alt;
    const download = $('download-reference');
    download.href = url;
    download.download = name;
    download.textContent = downloadLabel;
    referenceView.showModal();
    $('close-reference').focus();
  }

  function openResultView(event) {
    if (!outputUrl || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    if (event.currentTarget === $('output') && $('image-frame').dataset.comparing === 'true') return;
    event.preventDefault();
    openImageView({
      url: outputUrl, name: $('download').download, label: $('result-heading').textContent,
      details: $('result-summary').textContent, alt: $('output').alt, downloadLabel: 'Download PNG',
      prompt: $('output-prompt').textContent
    });
  }
  $('open-image').addEventListener('click', openResultView);
  $('output').addEventListener('click', openResultView);

  $('close-reference').addEventListener('click', () => referenceView.close());
  referenceView.addEventListener('close', () => {
    if (referenceView.open) return;
    $('full-reference').removeAttribute('src');
    $('download-reference').removeAttribute('href');
  });

  const promptKey = 'vigenflow.saved-prompts.v1';
  const editor = $('prompt-editor');
  const presetDialog = $('preset-dialog');
  const promptField = $('prompt');
  const picker = $('prompt-picker');
  let savedPrompts = [];
  let editingCommand = null;
  let autoCommand = false;
  let slashMatch = null;
  let matches = [];
  let activeOption = 0;
  let storageError = '';

  function readPrompts() {
    try {
      const entries = JSON.parse(localStorage.getItem(promptKey) || '[]');
      if (!Array.isArray(entries) || entries.some(item => !item || typeof item.name !== 'string' ||
          typeof item.command !== 'string' || typeof item.content !== 'string')) throw new Error();
      savedPrompts = entries;
      storageError = '';
    } catch {
      storageError = 'Could not read saved prompts.';
    }
  }

  function savePrompts(change) {
    readPrompts();
    if (storageError) {
      $(presetDialog.open ? 'preset-status' : 'library-status').textContent = storageError;
      return false;
    }
    try {
      const updated = change(savedPrompts);
      localStorage.setItem(promptKey, JSON.stringify(updated));
      savedPrompts = updated;
      return true;
    } catch (error) {
      $(presetDialog.open ? 'preset-status' : 'library-status').textContent = error.name === 'QuotaExceededError'
        ? 'Storage is full.' : error.message || 'Could not save.';
      return false;
    }
  }

  function closeEditor() {
    presetDialog.close();
  }

  function editPrompt(item, draft = '') {
    editor.reset();
    editingCommand = item?.command ?? null;
    autoCommand = !item;
    $('template-name').value = item?.name ?? '';
    $('template-command').value = item ? '/' + item.command : '';
    $('template-content').value = item?.content ?? draft;
    $('preset-status').textContent = '';
    $('preset-title').textContent = item ? 'Edit prompt' : 'New prompt';
    presetDialog.showModal();
    $('template-name').focus();
  }

  function renderPrompts() {
    const list = $('prompt-list');
    list.replaceChildren();
    if (!savedPrompts.length) list.textContent = 'No saved prompts.';
    for (const item of savedPrompts) {
      const row = document.createElement('article');
      row.className = 'prompt-row';
      const title = document.createElement('strong');
      title.textContent = item.name + ' ';
      const command = document.createElement('code');
      command.textContent = '/' + item.command;
      const body = document.createElement('p');
      body.textContent = item.content;
      const actions = document.createElement('div');
      actions.className = 'prompt-actions';
      const edit = document.createElement('button');
      const remove = document.createElement('button');
      edit.type = remove.type = 'button';
      edit.textContent = 'Edit';
      remove.textContent = 'Delete';
      edit.addEventListener('click', () => editPrompt(item));
      remove.addEventListener('click', () => {
        if (!confirm('Delete /' + item.command + '?')) return;
        if (!savePrompts(items => items.filter(entry => entry.command !== item.command))) return;
        if (editingCommand === item.command && presetDialog.open) closeEditor();
        renderPrompts();
        $('library-status').textContent = 'Prompt deleted.';
      });
      actions.append(edit, remove);
      row.append(title, command, body, actions);
      list.append(row);
    }
  }

  const commandField = $('template-command');
  commandField.addEventListener('input', () => { autoCommand = false; });
  $('template-name').addEventListener('input', event => {
    if (!autoCommand) return;
    const command = event.target.value.normalize('NFKC').toLowerCase()
      .replace(/[^\p{L}\p{N}\p{M}_-]/gu, '').replace(/^[^\p{L}\p{N}]+/u, '').slice(0, 64)
      .replace(/[\uD800-\uDBFF]$/, '');
    commandField.value = command ? '/' + command : '';
  });

  editor.addEventListener('submit', event => {
    event.preventDefault();
    const name = $('template-name').value.trim();
    const command = $('template-command').value.trim().replace(/^\//, '').normalize('NFKC').toLowerCase();
    const content = $('template-content').value.trim();
    if (!name || !content || command.length > 64 || !/^[\p{L}\p{N}][\p{L}\p{N}\p{M}_-]*$/u.test(command)) {
      $('preset-status').textContent = 'Enter a name, text and a command (letters, numbers, _ or -).';
      return;
    }
    if (!savePrompts(items => {
      if (items.some(item => item.command === command && item.command !== editingCommand)) {
        throw new Error('This slash command already exists.');
      }
      return [...items.filter(item => item.command !== editingCommand), { name, command, content }];
    })) return;
    renderPrompts();
    closeEditor();
    const message = 'Saved /' + command + '.';
    if ($('prompt-library').hidden) status(message);
    else $('library-status').textContent = message;
  });

  $('new-prompt').addEventListener('click', () => editPrompt());
  $('save-preset').addEventListener('click', () => {
    const model = selectedModel();
    editPrompt(null, model?.id === qwenEdit ? promptForModel(model) : promptField.value);
  });
  $('cancel-prompt').addEventListener('click', closeEditor);
  function showWorkspace() {
    const inLibrary = location.hash === '#prompts';
    $('prompt-library').hidden = !inLibrary;
    $('generation-form').hidden = inLibrary;
    $('prompts-link').setAttribute('aria-current', inLibrary ? 'page' : 'false');
    $('studio-link').setAttribute('aria-current', inLibrary ? 'false' : 'page');
    if (inLibrary) {
      readPrompts();
      renderPrompts();
      $('library-status').textContent = storageError;
    }
  }

  function hidePicker() {
    picker.hidden = true;
    slashMatch = null;
    promptField.removeAttribute('aria-activedescendant');
  }

  function selectOption(index) {
    activeOption = (index + matches.length) % matches.length;
    Array.from(picker.children).forEach((option, i) => option.setAttribute('aria-selected', String(i === activeOption)));
    const option = picker.children[activeOption];
    promptField.setAttribute('aria-activedescendant', option.id);
    option.scrollIntoView({ block: 'nearest' });
  }

  function replaceSlash(range, content) {
    if (promptField.value !== range.value) return;
    promptField.setRangeText(presetContent(content), range.start, range.end);
    promptField.focus();
  }

  function insertPrompt(item) {
    const range = slashMatch;
    hidePicker();
    if (!range || promptField.value !== range.value) return;
    if (item.reference) insertReference(range, item.reference);
    else showPromptFields(item, range);
  }

  function updatePicker() {
    const value = promptField.value;
    const caret = promptField.selectionStart;
    const match = value.slice(0, caret).match(/(?:^|\s)([\/@])([\p{L}\p{N}\p{M}_-]*)$/u);
    if (!match || caret !== promptField.selectionEnd) return hidePicker();
    const [, trigger, query] = match;
    const word = query.normalize('NFKC').toLowerCase();
    if (trigger === '@') {
      matches = selectedModel()?.edits_image ? activeReferences().map((reference, index) => ({
        reference, name: reference.name, tag: 'Image ' + (index + 1)
      })).filter(item => (item.tag + ' ' + item.name).toLowerCase().includes(word)) : [];
    } else {
      readPrompts();
      matches = savedPrompts.filter(item => item.command.includes(word) || item.name.toLowerCase().includes(word));
    }
    if (!matches.length) return hidePicker();
    const suffix = value.slice(caret).match(/^[\p{L}\p{N}\p{M}_-]*/u)[0];
    slashMatch = { value, start: caret - query.length - 1, end: caret + suffix.length };
    picker.replaceChildren(...matches.map((item, index) => {
      const option = document.createElement('button');
      option.type = 'button';
      option.id = 'suggestion-' + index;
      option.tabIndex = -1;
      option.setAttribute('role', 'option');
      option.textContent = (item.reference ? item.tag : '/' + item.command) + (item.name ? ' · ' + item.name : '');
      option.addEventListener('mousedown', event => event.preventDefault());
      option.addEventListener('click', () => insertPrompt(item));
      return option;
    }));
    picker.hidden = false;
    selectOption(0);
  }

  promptField.addEventListener('input', updatePicker);
  promptField.addEventListener('click', updatePicker);
  document.addEventListener('pointerdown', event => {
    if (event.target !== promptField && !picker.contains(event.target)) hidePicker();
  });
  promptField.addEventListener('keydown', event => {
    if (event.isComposing || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === 'Enter' && (picker.hidden || event.shiftKey)) {
      event.preventDefault();
      replaceEditor('\n', promptField.selectionStart, promptField.selectionEnd);
      hidePicker();
      return;
    }
    if (picker.hidden) return;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      selectOption(activeOption + (event.key === 'ArrowDown' ? 1 : -1));
    } else if (event.key === 'Enter') {
      event.preventDefault();
      insertPrompt(matches[activeOption]);
    } else if (event.key === 'Escape' || event.key === 'Tab' || event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      hidePicker();
    }
  });

  window.addEventListener('hashchange', showWorkspace);
  showWorkspace();
  const fieldsDialog = $('prompt-fields');
  const fieldsForm = $('prompt-fields-form');
  const fieldPattern = /\{\{\s*([\p{L}_][\p{L}\p{N}_-]*(?:[ \t]+[\p{L}\p{N}_-]+)*)\s*(?:\|\s*(text|textarea)((?::(?:required|(?:label|placeholder|default)="(?:[^"\\]|\\.)*"))*))?\s*\}\}/gu;
  let pendingPrompt;

  function showPromptFields(item, range) {
    const fields = new Map();
    for (const [, name, type, settings = ''] of item.content.matchAll(fieldPattern)) {
      const props = fields.get(name) || {};
      if (type) props.type = type;
      for (const [, key, value] of settings.matchAll(/:(required|label|placeholder|default)(?:="((?:[^"\\]|\\.)*)")?/g)) {
        props[key] = value === undefined ? true : value.replace(/\\(["\\])/g, '$1');
      }
      fields.set(name, props);
    }
    if (!fields.size) return replaceSlash(range, item.content);

    const controls = new Map();
    const rows = [];
    for (const [name, props] of fields) {
      const label = document.createElement('label');
      label.textContent = (props.label || name.replaceAll('_', ' ')) + (props.required ? ' *' : '');
      const field = document.createElement(props.type === 'textarea' ? 'textarea' : 'input');
      field.required = Boolean(props.required);
      field.placeholder = props.placeholder || '';
      field.value = props.default || '';
      field.addEventListener('input', () => field.setCustomValidity(''));
      label.append(field);
      controls.set(name, field);
      rows.push(label);
    }
    $('prompt-fields-title').textContent = item.name;
    $('variable-fields').replaceChildren(...rows);
    pendingPrompt = { item, range, controls };
    fieldsDialog.showModal();
    rows[0].lastElementChild.focus();
  }

  fieldsForm.addEventListener('submit', event => {
    event.preventDefault();
    const { item, range, controls } = pendingPrompt;
    for (const field of controls.values()) {
      if (field.required && !field.value.trim()) {
        field.setCustomValidity('Fill in this field.');
        field.reportValidity();
        return;
      }
    }
    const content = item.content.replace(fieldPattern, (_, name) => controls.get(name).value);
    fieldsDialog.close();
    replaceSlash(range, content);
  });
  $('cancel-fields').addEventListener('click', () => fieldsDialog.close());
  fieldsDialog.addEventListener('close', () => {
    if (fieldsDialog.open) return;
    pendingPrompt = null;
    promptField.focus();
  });

  const dropPanel = $('controls').closest('.settings');
  const dropHint = $('drop-hint');
  const canUpload = () => !$('generation-form').hidden && !$('controls').disabled &&
    !$('input-image').disabled && !document.querySelector('dialog[open]');
  const hasFiles = event => event.dataTransfer.types.includes('Files');
  const hideDrop = () => { dropHint.hidden = true; };

  dropPanel.addEventListener('dragover', event => {
    if (!hasFiles(event) || !canUpload()) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
    dropHint.hidden = false;
  });
  dropPanel.addEventListener('dragleave', event => {
    if (!dropPanel.contains(event.relatedTarget)) hideDrop();
  });
  dropPanel.addEventListener('drop', event => {
    if (!hasFiles(event) || !canUpload()) return;
    event.preventDefault();
    const files = [...event.dataTransfer.files];
    if (files.length) addReferences(files);
  });
  document.addEventListener('dragover', event => {
    if (hasFiles(event)) event.preventDefault();
  });
  document.addEventListener('drop', event => {
    if (hasFiles(event)) event.preventDefault();
    hideDrop();
  });
  window.addEventListener('dragend', hideDrop);
  window.addEventListener('blur', hideDrop);

  document.addEventListener('paste', event => {
    if (!canUpload()) return;
    const clipboard = event.clipboardData;
    const files = Array.from(clipboard?.files || []).filter(item => item.type.startsWith('image/'));
    if (!files.length) return;
    event.preventDefault();
    addReferences(files);
  });

  const root = document.documentElement;
  const themeToggle = $('theme-toggle');
  function showTheme() {
    themeToggle.setAttribute('aria-checked', root.dataset.theme === 'light');
  }
  themeToggle.addEventListener('click', () => {
    root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light';
    try { localStorage.setItem('vgf-theme', root.dataset.theme); } catch {}
    showTheme();
  });
  showTheme();
  composer.addEventListener('paste', event => {
    if (event.clipboardData.files.length) return;
    event.preventDefault();
    const text = event.clipboardData.getData('text/plain').replace(/\r\n?/g, '\n');
    replaceEditor(text, composer.selectionStart, composer.selectionEnd);
    updatePicker();
  });
  loadModels();
})();
</script>
</body>
</html>
)VGFHTML";

}  // namespace server
