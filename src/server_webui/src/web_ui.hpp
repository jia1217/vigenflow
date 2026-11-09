#pragma once

namespace server {

// Embedded so the browser UI travels with vgf-serve on Ubuntu and Windows.
// No web server, JavaScript build, or external assets are required.
inline constexpr char kWebUi[] = R"VGFHTML(<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>VigenFlow Image Client</title>
  <style>
    :root { color-scheme: light; --ink: #182235; --muted: #5e6b7e; --line: #dfe5ed; --accent: #4355d1; --soft: #eef1ff; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f5f7fb; color: var(--ink); font: 15px/1.5 system-ui, sans-serif; }
    main { max-width: 1680px; margin: 0 auto; padding: 28px 32px 48px; }
    .app-header { display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-bottom: 26px; }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-mark { display: grid; place-items: center; width: 42px; height: 42px; border-radius: 12px; background: var(--accent); color: white; font-size: 24px; font-weight: 750; }
    h1 { margin: 0; font-size: 25px; letter-spacing: -.7px; line-height: 1.2; }
    .intro { margin: 4px 0 0; font-size: 13px; color: var(--muted); }
    .badge { display: inline-flex; align-items: center; gap: 7px; padding: 5px 10px; border-radius: 20px; background: #edf0f5; color: var(--muted); font-size: 12px; font-weight: 600; }
    #connection-state::before { content: ''; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
    #connection-state[data-state="connected"] { background: #e5f4eb; color: #246641; }
    #connection-state[data-state="error"] { background: #fceae8; color: #9d322c; }
    .workspace { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 3fr); gap: 24px; align-items: start; }
    .settings { min-width: 0; position: sticky; top: 24px; }
    .controls-layout { display: grid; grid-template-rows: minmax(0, 1fr) auto; max-height: calc(100vh - 138px); }
    .panel { min-width: 0; background: #fff; border: 1px solid var(--line); border-radius: 16px; box-shadow: 0 3px 16px #1a284007; }
    .form-content { padding: 24px 24px 0; min-height: 0; overflow: auto; scrollbar-width: thin; }
    h2, legend { padding: 0; font-size: 16px; font-weight: 700; margin: 0 0 16px; }
    .section { border-top: 1px solid var(--line); margin-top: 22px; padding-top: 22px; }
    label { display: block; font-size: 13px; font-weight: 650; margin-bottom: 7px; }
    input, textarea, select, button, .download, .secondary-link {
      max-width: 100%; border: 1px solid #cbd3df; border-radius: 8px; padding: 10px 12px;
      font: inherit; color: inherit; background: #fff;
    }
    input, textarea, select { width: 100%; min-width: 0; }
    input, select, button, .download, .secondary-link { min-height: 44px; }
    textarea { min-height: 140px; resize: vertical; line-height: 1.65; }
    input[readonly], select:disabled { background: #f4f6f9; color: var(--muted); }
    input[type="checkbox"] { width: 17px; height: 17px; min-height: 0; flex-shrink: 0; accent-color: var(--accent); }
    button, .download, .secondary-link { cursor: pointer; text-align: center; font-weight: 600; font-size: 13px; text-decoration: none; }
    button:hover:not(:disabled), .secondary-link:hover { background: #f1f4fa; border-color: #abb8cc; }
    button:disabled { cursor: not-allowed; opacity: .55; }
    :focus-visible { outline: 3px solid #818fee; outline-offset: 3px; }
    fieldset { margin: 0; padding: 0; border: 0; min-width: 0; }
    .field { margin-bottom: 18px; min-width: 0; }
    .grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 0 16px; }
    .model-row, .seed-row { display: flex; gap: 8px; }
    .model-row select, .seed-row input { min-width: 0; flex: 1; }
    .model-row button, .seed-row button { flex-shrink: 0; }
    .help { display: block; margin: 7px 0 0; color: var(--muted); font-size: 12px; line-height: 1.6; }
    .custom-size { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 10px; margin-top: 12px; }
    .custom-size label { color: var(--muted); font-size: 12px; }
    .advanced { border-top: 1px solid var(--line); margin-top: 22px; }
    summary { cursor: pointer; font-weight: 650; font-size: 13px; padding: 15px 0; }
    .advanced[open] > summary { margin-bottom: 4px; }
    .generate-bar { flex-shrink: 0; padding: 16px 24px; border-top: 1px solid var(--line); background: #fff; border-radius: 0 0 16px 16px; box-shadow: 0 -6px 16px #18223506; }
    #generate, .download { background: var(--accent); color: white; border-color: var(--accent); }
    #generate { width: 100%; min-height: 48px; font-size: 15px; }
    #generate:hover:not(:disabled), .download:hover { background: #3445b7; border-color: #3445b7; }
    #generate:disabled { cursor: wait; }
    #status { margin-top: 10px; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    #status[data-error="true"], #lora-error { color: #a12a24; }
    #retry { width: 100%; margin-top: 12px; }
    .preview-panel { position: sticky; top: 24px; max-height: calc(100vh - 48px); overflow: auto; padding: 24px; }
    .prompt-field { padding-bottom: 20px; margin-bottom: 20px; border-bottom: 1px solid var(--line); }
    .prompt-field textarea { min-height: 110px; }
    .preview-heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 18px; }
    .preview-heading h2 { margin: 0; }
    .preview-placeholder { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px;
      min-height: clamp(240px, 40vh, 480px); padding: 24px; border: 1px dashed #cbd3df; border-radius: 12px;
      background: #f8f9fc; text-align: center; color: var(--muted); }
    .preview-placeholder svg { width: 48px; height: 48px; margin-bottom: 12px; color: #8b98ae; }
    .preview-placeholder strong { font-size: 17px; color: var(--ink); }
    .preview-placeholder p { margin: 0; max-width: 300px; font-size: 13px; }
    .preview-panel[data-busy="true"] #preview-state { background: var(--soft); color: var(--accent); }
    .preview-panel[data-busy="true"] #preview-state::before { content: ''; width: 10px; height: 10px; border: 2px solid #b6bff4; border-top-color: var(--accent); border-radius: 50%; animation: spin 1s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .image-stage { display: grid; place-items: center; border-radius: 10px; background: #f4f6fa; padding: 12px; }
    #output { display: block; width: 100%; height: auto; max-width: 100%; max-height: 62vh; object-fit: contain; border-radius: 6px; }
    .image-frame { position: relative; width: 100%; max-width: calc(62vh * var(--image-ratio, 1)); overflow: hidden; border-radius: 6px; --split: 50%; }
    .image-frame[data-comparing="true"] { cursor: ew-resize; touch-action: pan-y; user-select: none; }
    .compare-before { position: absolute; inset: 0; clip-path: inset(0 calc(100% - var(--split)) 0 0); pointer-events: none; }
    #before-image { display: block; width: 100%; height: 100%; object-fit: cover; object-position: center; }
    .compare-divider { position: absolute; top: 0; bottom: 0; left: var(--split); width: 2px; transform: translateX(-50%); background: white; box-shadow: 0 0 4px #18223580; pointer-events: none; }
    .compare-divider span { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); display: grid; place-items: center;
      width: 38px; height: 38px; border: 2px solid white; border-radius: 50%; background: var(--accent); color: white; font-size: 21px; line-height: 1; box-shadow: 0 2px 8px #18223540; }
    .comparison-label { position: absolute; top: 10px; padding: 4px 9px; border-radius: 6px; background: #182235c9; color: white; font-size: 12px; font-weight: 650; pointer-events: none; }
    #before-label { left: 10px; }
    #after-label { right: 10px; }
    .comparison-tools { display: flex; margin-bottom: 12px; }
    .view-switch { display: inline-flex; gap: 4px; padding: 4px; border: 1px solid var(--line); border-radius: 10px; background: #f4f6fa; }
    .view-switch button { min-height: 36px; padding: 6px 12px; background: transparent; border-color: transparent; color: var(--muted); }
    .view-switch button[aria-pressed="true"] { background: white; border-color: #cbd3df; color: var(--accent); box-shadow: 0 1px 3px #1822350d; }
    .comparison-control { margin-top: 12px; }
    .comparison-control label { margin-bottom: 0; }
    #compare-position { display: block; padding: 0; border: 0; background: transparent; accent-color: var(--accent); cursor: ew-resize; }
    .comparison-control .help { margin-top: 0; }
    #input-preview { display: block; max-width: 100%; max-height: 160px; margin-top: 12px; border-radius: 8px; }
    .result-toolbar { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
    .download, .secondary-link { display: inline-flex; align-items: center; justify-content: center; }
    #result-summary { margin: 14px 0 0; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .result-details { margin-top: 16px; border-top: 1px solid var(--line); }
    #details { margin: 0; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    #details dt { color: var(--muted); font-size: 12px; }
    #details > div, #output-prompt { min-width: 0; overflow-wrap: anywhere; }
    #details dd { margin: 3px 0 0; font-size: 13px; }
    #output-prompt { font-size: 13px; white-space: pre-wrap; }
    #lora-field legend { margin-bottom: 8px; font-size: 13px; }
    #lora-list { display: grid; gap: 6px; }
    .lora { position: relative; border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
    .lora:has(input:checked) { border-color: #8d9ae9; background: #f6f7ff; }
    .lora[data-missing="true"] { background: #f6f7f9; color: var(--muted); }
    .lora-row { display: flex; align-items: center; gap: 8px; padding-right: 64px; }
    .lora label { display: flex; flex: 1; min-width: 0; min-height: 32px; align-items: center; gap: 8px; margin: 0; cursor: pointer; }
    .lora-name { min-width: 0; overflow-wrap: anywhere; font-size: 13px; }
    .lora .badge { font-size: 10px; padding: 3px 7px; margin-bottom: 7px; }
    .lora-trigger { margin: 0 0 7px; font-size: 12px; color: var(--muted); overflow-wrap: anywhere; }
    .lora-add { flex-shrink: 0; padding: 5px 8px; min-height: 32px; font-size: 12px; color: var(--accent); }
    .lora summary { position: absolute; top: 8px; right: 10px; min-height: 32px; padding: 7px 0;
                    font-size: 11px; font-weight: 500; color: var(--muted); }
    .lora-info { margin-top: 8px; padding-top: 10px; border-top: 1px solid var(--line); }
    .lora small { display: block; font-size: 11px; color: var(--muted); overflow-wrap: anywhere; }
    .lora-help { margin: 7px 0 0; }
    [hidden] { display: none !important; }
    @media (max-width: 1000px) {
      .workspace { grid-template-columns: minmax(0, 1fr); }
      .preview-panel { position: static; max-height: none; overflow: visible; }
      .settings { position: static; }
      .controls-layout { display: block; max-height: none; }
      .form-content { overflow: visible; }
    }
    @media (max-width: 600px) {
      main { padding: 18px 12px 32px; }
      .app-header { align-items: flex-start; gap: 10px; }
      .brand-mark { display: none; }
      h1 { font-size: 23px; }
      .intro { max-width: 190px; }
      .form-content { padding: 18px 18px 0; }
      .preview-panel { padding: 18px; }
      .generate-bar { padding: 14px 18px; }
      .grid { grid-template-columns: minmax(0, 1fr); }
      .model-row { flex-wrap: wrap; }
      .model-row select { flex-basis: 100%; }
      #refresh { width: 100%; }
      .preview-heading { flex-wrap: wrap; }
    }
    @media (prefers-reduced-motion: reduce) { *, *::before { animation: none !important; scroll-behavior: auto !important; } }
  </style>
</head>
<body>
<main>
  <header class="app-header">
    <div class="brand">
      <span class="brand-mark" aria-hidden="true">V</span>
      <div><h1>VigenFlow</h1><p class="intro">Create and edit images on your device.</p></div>
    </div>
    <span id="connection-state" class="badge" data-state="connecting">Connecting</span>
  </header>
  <noscript>This image client needs JavaScript enabled in your browser.</noscript>
  <form id="generation-form" class="workspace">
    <div class="settings panel">
        <fieldset id="controls" disabled>
          <div class="controls-layout">
          <div class="form-content">
            <div class="field">
              <label for="model">Model</label>
              <div class="model-row">
                <select id="model" required><option value="">Loading models…</option></select>
                <button id="refresh" type="button" aria-label="Refresh models">Refresh</button>
              </div>
            </div>
            <fieldset id="lora-field" class="field" hidden>
              <legend>Styles (LoRAs)</legend>
              <div id="lora-list"></div>
              <p class="help lora-help">Optional. Add words also selects the style.</p>
              <small id="lora-error" class="help" hidden></small>
            </fieldset>
            <div id="input-field" class="field" hidden>
              <label for="input-image">Image to edit</label>
              <input id="input-image" type="file" accept="image/png,image/jpeg,image/bmp,image/gif" aria-describedby="input-help">
              <small id="input-help" class="help">PNG, JPEG, BMP or GIF, up to 40 MB. Center-cropped to the selected aspect ratio and resized to the selected image size.</small>
              <img id="input-preview" alt="Selected input image" hidden>
            </div>
            <section class="section" aria-labelledby="settings-heading">
              <h2 id="settings-heading">Image settings</h2>
              <div class="grid">
                <div class="field">
                  <label for="size">Image size</label>
                  <select id="size" aria-describedby="size-help"><option value="1024x1024">1024 × 1024</option></select>
                  <div id="custom-size" class="custom-size" hidden>
                    <div><label for="width">Width (px)</label><input id="width" type="number" value="1024" required disabled></div>
                    <div><label for="height">Height (px)</label><input id="height" type="number" value="1024" required disabled></div>
                  </div>
                  <small id="size-help" class="help">Sizes supported by this model.</small>
                </div>
                <div class="field">
                  <label for="seed">Seed</label>
                  <div class="seed-row">
                    <input id="seed" type="number" min="-2147483648" max="2147483647" step="1" value="42" required aria-describedby="seed-help">
                    <button id="random-seed" type="button" aria-label="Choose a random seed">Random</button>
                  </div>
                  <small id="seed-help" class="help">Keep the seed to repeat a result. Random picks a new one.</small>
                </div>
              </div>
            </section>
            <details id="advanced-settings" class="advanced">
              <summary>Advanced settings</summary>
              <div class="grid">
                <div class="field">
                  <label for="steps">Generation steps</label>
                  <select id="steps" required aria-describedby="steps-help"></select>
                  <small id="steps-help" class="help">Fixed for this model.</small>
                </div>
                <div class="field">
                  <label for="response-format">Image delivery</label>
                  <select id="response-format" aria-describedby="format-help"><option value="url">URL (default)</option><option value="b64_json">Base64</option></select>
                  <small id="format-help" class="help">Both options display and download a PNG.</small>
                </div>
              </div>
              
              <div class="field">
                <label for="base-url">Server address</label>
                <input id="base-url" type="url" readonly aria-describedby="connection-help">
                <small id="connection-help" class="help">This page uses the server that hosts it.</small>
              </div>
            </details>
          </div>
          <div class="generate-bar">
            <button id="generate" type="submit">Generate image</button>
            <div id="status" role="status" aria-live="polite">Connecting to your server…</div>
          </div>
          </div>
        </fieldset>
      <button id="retry" type="button" hidden>Retry connection</button>
    </div>
    <section id="preview-panel" class="preview-panel panel" aria-labelledby="result-heading">
      <div class="field prompt-field">
        <label id="prompt-label" for="prompt">Describe your image</label>
        <textarea id="prompt" required disabled aria-describedby="prompt-help" placeholder="Describe the subject, setting, lighting, and style…">A cinematic mountain village at sunset, snow, warm window lights, detailed, realistic</textarea>
        <small id="prompt-help" class="help">Include the subject, setting, and style. Ctrl / ⌘ + Enter to generate.</small>
      </div>
      <div class="preview-heading">
        <h2 id="result-heading">Image preview</h2>
        <span id="preview-state" class="badge">No image yet</span>
      </div>
      <div id="preview-placeholder" class="preview-placeholder">
        <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><rect x="5" y="5" width="38" height="38" rx="7"/><circle cx="16" cy="16" r="4"/><path d="M6 35l11-11 8 8 6-6 12 12"/></svg>
        <strong id="preview-message">Your next image starts here</strong>
        <p id="preview-hint">Describe what you have in mind, choose your settings, and select Generate image.</p>
      </div>
      <div id="result" hidden>
        <div id="comparison-tools" class="comparison-tools" hidden>
          <div class="view-switch" role="group" aria-label="Image view">
            <button id="view-edited" type="button" aria-pressed="true">Edited image</button>
            <button id="view-compare" type="button" aria-pressed="false">Before / after</button>
          </div>
        </div>
        <div class="image-stage">
          <div id="image-frame" class="image-frame">
            <img id="output" alt="Generated image" draggable="false">
            <div id="compare-before" class="compare-before" hidden><img id="before-image" alt="Input image before editing, cropped to match the result" draggable="false"></div>
            <div id="compare-divider" class="compare-divider" aria-hidden="true" hidden><span>↔</span></div>
            <span id="before-label" class="comparison-label" aria-hidden="true" hidden>Before</span>
            <span id="after-label" class="comparison-label" aria-hidden="true" hidden>After</span>
          </div>
        </div>
        <div id="comparison-control" class="comparison-control" hidden>
          <label for="compare-position">Slide to compare</label>
          <input id="compare-position" type="range" min="0" max="100" value="50" aria-describedby="comparison-help">
          <small id="comparison-help" class="help">Drag the divider or use the slider’s arrow keys. Before shows the centered crop used for editing.</small>
        </div>
        <p id="comparison-unavailable" class="help" hidden>Before / after is unavailable because this browser could not display the input image.</p>
        <div class="result-toolbar">
          <a id="download" class="download" download="vigenflow.png">Download PNG</a>
          <a id="open-image" class="secondary-link" target="_blank" rel="noopener">View full size</a>
        </div>
        <p id="result-summary"></p>
        <details class="result-details">
          <summary>Generation details</summary>
          <dl id="details"></dl>
          <p id="output-prompt"></p>
        </details>
      </div>
    </section>
  </form>
</main>
<script>
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const api = new URL('/v1/', location.origin);
  $('base-url').value = api.href.replace(/\/$/, '');
  let models = [];
  let outputUrl = '';
  let previewUrl = '';
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

  // A separate URL keeps the submitted input paired with its result when the
  // file picker changes. Decoding failure must not discard a successful edit.
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
    $('prompt-label').textContent = model.edits_image ? 'Describe your changes' : 'Describe your image';
    $('prompt-help').textContent = model.edits_image
      ? 'Describe what to change in your input image. Ctrl / ⌘ + Enter to edit.'
      : 'Include the subject, setting, and style. Ctrl / ⌘ + Enter to generate.';
    if (!outputUrl) {
      $('preview-message').textContent = model.edits_image ? 'Make an image your own' : 'Your next image starts here';
      $('preview-hint').textContent = model.edits_image
        ? 'Choose an input image, describe your changes, and select Edit image.'
        : 'Describe what you have in mind, choose your settings, and select Generate image.';
    }
  }

  function status(message, error = false) {
    $('status').textContent = message;
    $('status').dataset.error = String(error);
  }

  async function readResponse(response) {
    let data;
    try { data = await response.json(); }
    catch { throw new Error(`The server returned an invalid response (HTTP ${response.status}).`); }
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
    return `${w} × ${h} (${w === h ? 'square' : w > h ? 'landscape' : 'portrait'})`;
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
      ? `Custom sizes: ${limits.min}–${limits.max} pixels per side, in multiples of ${limits.multiple}.`
      : 'Sizes supported by this model.';
    updateCustomSize();
  }

  function updateCustomSize() {
    const custom = $('size').value === 'custom';
    $('custom-size').hidden = !custom;
    $('width').disabled = $('height').disabled = !custom;
  }

  function chosenSize(model) {
    if ($('size').value !== 'custom') return $('size').value;
    const limits = model.size_limits;
    const sides = [Number($('width').value), Number($('height').value)];
    if (sides.some(side => !Number.isInteger(side) || side < limits.min || side > limits.max || side % limits.multiple)) {
      throw new Error(`Width and height must each be ${limits.min}–${limits.max} pixels, in multiples of ${limits.multiple}.`);
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
        add.setAttribute('aria-label', `Add ${lora.name} words to prompt and select this style`);
        add.disabled = box.disabled;
        add.addEventListener('click', () => {
          const words = lora.trigger.split(',').map(word => word.trim()).filter(Boolean);
          const existing = $('prompt').value.toLocaleLowerCase();
          const missing = words.filter(word => !existing.includes(word.toLocaleLowerCase()));
          if (missing.length) $('prompt').value = [$('prompt').value.trim(), ...missing].filter(Boolean).join(', ');
          box.checked = true;
          $('prompt').focus();
          status(`${lora.name} selected. Its prompt words are included.`);
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

  function updateModel() {
    const model = selectedModel();
    if (!model) return;
    fillOptions($('steps'), model.supported_steps, model.default_steps);
    $('steps').disabled = model.supported_steps.length < 2;
    $('steps-help').hidden = model.supported_steps.length > 1;
    fillSizes(model);
    fillLoras(model);
    $('seed').value = model.default_seed;
    $('input-field').hidden = !model.edits_image;
    $('input-image').required = model.edits_image;
    $('input-image').disabled = !model.edits_image;
    updateAction(model);
  }

  async function loadModels() {
    const previous = $('model').value;
    $('controls').disabled = $('prompt').disabled = true;
    $('retry').hidden = true;
    setConnection('connecting', 'Connecting');
    status('Connecting to your server…');
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    try {
      const data = await readResponse(await fetch(new URL('models', api), { signal: controller.signal, cache: 'no-store' }));
      if (!Array.isArray(data.data) || !data.data.length) throw new Error('No models are available on this server.');
      models = data.data;
      $('model').replaceChildren(...models.map(model => new Option(
        model.id + (model.edits_image ? ' (image editing)' : ''), model.id)));
      const preferred = models.find(model => model.id === previous) || models.find(model => model.is_default) || models[0];
      $('model').value = preferred.id;
      updateModel();
      $('controls').disabled = $('prompt').disabled = false;
      setConnection('connected', 'Connected');
      status('Ready to create.');
    } catch (error) {
      setConnection('error', 'Disconnected');
      status(`Cannot load models. ${error.message} Check that vgf-serve is running.`, true);
      $('retry').hidden = false;
    } finally { clearTimeout(timeout); }
  }

  $('random-seed').addEventListener('click', () => {
    const random = crypto.getRandomValues(new Uint32Array(1))[0] & 0x7fffffff;
    const previous = Number($('seed').value);
    $('seed').value = random === previous ? (random + 1) % 2147483648 : random;
    status('New seed selected. Generate an image to try it.');
  });
  $('generation-form').addEventListener('keydown', event => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey) && !event.isComposing) {
      event.preventDefault();
      if (!$('controls').disabled) $('generation-form').requestSubmit();
    }
  });
  // Reveal invalid advanced fields before the browser tries to focus them.
  $('generation-form').addEventListener('invalid', event => {
    let section = event.target.closest('details');
    while (section) { section.open = true; section = section.parentElement.closest('details'); }
    status(event.target.validationMessage, true);
  }, true);
  $('model').addEventListener('change', updateModel);
  $('size').addEventListener('change', updateCustomSize);
  $('refresh').addEventListener('click', loadModels);
  $('retry').addEventListener('click', loadModels);
  $('input-image').addEventListener('change', () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const file = $('input-image').files[0];
    previewUrl = file ? URL.createObjectURL(file) : '';
    $('input-preview').hidden = !file;
    if (file) $('input-preview').src = previewUrl;
    else $('input-preview').removeAttribute('src');
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
    const prompt = $('prompt').value.trim();
    if (!prompt || new TextEncoder().encode(prompt).length > 16384 || prompt.includes('\0')) {
      status('Enter a prompt of 1–16384 UTF-8 bytes, without NUL characters.', true);
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
    // A form sends the names comma-separated; an empty value means no LoRA.
    if (model.loras) params.loras = [...$('lora-list').querySelectorAll('input:checked')].map(box => box.value);
    let body;
    let editInput = null;
    const headers = { Accept: 'application/json' };
    if (model.edits_image) {
      const file = $('input-image').files[0];
      if (!file || file.size > 40 * 1024 * 1024) {
        status('Choose an input image up to 40 MB.', true);
        return;
      }
      body = new FormData();
      for (const [key, value] of Object.entries(params)) body.append(key, String(value));
      body.append('image', file);
      editInput = file;
    } else {
      headers['Content-Type'] = 'application/json';
      body = JSON.stringify(params);
    }
    $('controls').disabled = $('prompt').disabled = true;
    $('generation-form').setAttribute('aria-busy', 'true');
    const started = performance.now();
    const elapsed = () => (performance.now() - started) / 1000;
    $('preview-panel').dataset.busy = 'true';
    $('preview-panel').setAttribute('aria-busy', 'true');
    $('generate').textContent = model.edits_image ? 'Editing…' : 'Generating…';
    if (!outputUrl) {
      $('preview-message').textContent = model.edits_image ? 'Applying your changes…' : 'Creating your image…';
      $('preview-hint').textContent = 'The first image may take longer while the model loads.';
    }
    status(`${model.edits_image ? 'Editing' : 'Generating'} your image. The first run may take longer while the model loads.`);
    const progress = () => { $('preview-state').textContent = `${model.edits_image ? 'Editing' : 'Generating'} · ${elapsed().toFixed(0)} s`; };
    progress();
    const timer = setInterval(progress, 1000);
    let nextUrl = '';
    let nextBeforeUrl = '';
    try {
      const route = model.edits_image ? 'images/edits' : 'images/generations';
      const data = await readResponse(await fetch(new URL(route, api), { method: 'POST', headers, body }));
      const item = data.data?.[0];
      let blob;
      if (item?.b64_json) {
        const bytes = Uint8Array.from(atob(item.b64_json), char => char.charCodeAt(0));
        blob = new Blob([bytes], { type: 'image/png' });
      } else if (item?.url) {
        // Fetch our image path on the page's origin, including behind an HTTPS proxy.
        const url = new URL(item.url, location.origin);
        if (!url.pathname.startsWith('/images/')) throw new Error('The server returned an unexpected image URL.');
        const response = await fetch(url.pathname + url.search);
        if (!response.ok) throw new Error(`Cannot load the generated image (HTTP ${response.status}).`);
        blob = await response.blob();
      } else { throw new Error('The server did not return an image.'); }
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
      if (matchMedia('(max-width: 1000px)').matches) {
        $('preview-panel').scrollIntoView({ block: 'start', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
      }
    } catch (error) {
      clearInterval(timer);
      if (nextUrl) URL.revokeObjectURL(nextUrl);
      if (nextBeforeUrl) URL.revokeObjectURL(nextBeforeUrl);
      $('preview-state').textContent = outputUrl ? 'Previous image' : 'Try again';
      if (!outputUrl) {
        $('preview-message').textContent = 'Your image could not be created';
        $('preview-hint').textContent = 'Check the message below Generate image, then try again.';
      }
      status(`Image request failed: ${error.message}`, true);
    } finally {
      clearInterval(timer);
      $('controls').disabled = $('prompt').disabled = false;
      $('generation-form').removeAttribute('aria-busy');
      $('preview-panel').dataset.busy = 'false';
      $('preview-panel').removeAttribute('aria-busy');
      $('generate').textContent = model.edits_image ? 'Edit image' : 'Generate image';
    }
  });
  loadModels();
})();
</script>
</body>
</html>
)VGFHTML";

}  // namespace server
