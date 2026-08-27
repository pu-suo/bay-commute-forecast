// site/app.js
//
// The map draws two kinds of claim and keeps them visually distinct. Freeway
// centrelines are a forecast from a detector on that stretch of road: thick,
// opaque, scored on the accuracy page. Side streets are inferred from whatever
// freeway runs nearby: thin, translucent, and switchable.
//
// Colours are computed from the per-detector speed series rather than baked in,
// so moving the scrubber is an arithmetic pass over ~60k ways, not a refetch.

const SCALE = [
  [0.95, '#1f9d84'],
  [0.85, '#7cb342'],
  [0.70, '#e8a020'],
  [0.50, '#dd6a1f'],
  [0.00, '#c0392b'],
];
const NO_DATA = '#b9c2ce';
// Matches forecast.surface.ALPHA. Arterials absorb half the freeway's
// proportional slowdown; a stated prior, not a fitted value.
const ALPHA = 0.5;

// On GitHub Pages the static files and the API live on different origins, so
// the base is read from a config file the deploy writes. Missing or empty means
// same-origin, which is how the local server runs.
let API = '';
const api = path => API + path;

const $ = id => document.getElementById(id);
const fmtHM = d => d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
const colourFor = r => (SCALE.find(s => r >= s[0]) || SCALE[SCALE.length - 1])[1];

const state = {
  net: null, cor: null, geo: null, slot: 8, mode: 'leave',
  showSurface: true, layers: [], pending: null,
  points: { from: null, to: null },
};

// ---- map --------------------------------------------------------------------
// preferCanvas matters here: 60k ways as individual SVG nodes is a slideshow.
const map = L.map('map', { preferCanvas: true, zoomControl: true })
  .setView([37.72, -122.15], 10);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap, &copy; CARTO', maxZoom: 18,
}).addTo(map);
const routeLayer = L.layerGroup().addTo(map);
const pinLayer = L.layerGroup().addTo(map);

function drawLegend() {
  $('legend').innerHTML =
    '<span>jam</span>' + SCALE.slice().reverse().map(s => `<i style="background:${s[1]}"></i>`).join('') +
    '<span>free</span>';
}

// ---- colouring --------------------------------------------------------------
function ratioOf(station, slot) {
  const series = state.net.speeds[station];
  if (!series) return null;
  const v = series[slot];
  if (v == null || v < 0) return null;
  const ff = state.net.freeflow[station] || 65;
  return Math.min(Math.max(v / ff, 0.15), 1.05);
}

// The spread model, run in the browser on the weights the builder precomputed:
// local speed falls by ALPHA times the freeway's own proportional slowdown.
function surfaceFactor(weights, slot) {
  let num = 0, den = 0;
  for (const [station, w] of weights) {
    const r = ratioOf(station, slot);
    if (r == null) continue;
    num += w * r; den += w;
  }
  if (!den) return null;
  return 1 - ALPHA * (1 - num / den);
}

function drawNetwork() {
  const { freeway, surface } = state.geo;
  const slot = state.slot;
  state.layers.forEach(l => map.removeLayer(l));
  state.layers = [];

  const buckets = new Map();
  const push = (key, coords) => {
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(coords);
  };

  let measured = 0, neutral = 0;
  for (const seg of freeway) {
    const r = seg.s == null ? null : ratioOf(seg.s, slot);
    if (r == null) { push('n|' + NO_DATA, seg.c); neutral++; }
    else { push('f|' + colourFor(r), seg.c); measured++; }
  }
  if (state.showSurface) {
    for (const way of surface) {
      const f = surfaceFactor(way.w, slot);
      if (f == null) continue;
      push('s|' + colourFor(f), way.c);
    }
  }

  // one canvas polyline per (class, colour) instead of 60k layers
  const z = map.getZoom();
  for (const [key, lines] of buckets) {
    const [kind, colour] = key.split('|');
    const style = kind === 's'
      ? { color: colour, weight: z >= 12 ? 2 : 1.2, opacity: 0.45, lineCap: 'butt' }
      : kind === 'n'
        ? { color: colour, weight: 1.6, opacity: 0.6 }
        : { color: colour, weight: z >= 12 ? 5 : 3.4, opacity: 0.95 };
    const layer = L.polyline(lines, style);
    layer.addTo(map);
    if (kind === 's') layer.bringToBack();
    state.layers.push(layer);
  }

  const t = new Date(state.net.slots[slot]);
  $('slotLabel').textContent =
    t.toLocaleDateString([], { weekday: 'long' }) + ' ' + fmtHM(t);
  $('coverNote').textContent =
    `${measured.toLocaleString()} forecast stretches · ${neutral.toLocaleString()} with no detector`;
  renderCorridors();
}

// scrubbing fires faster than the map can draw; coalesce onto the frame
function requestDraw() {
  if (state.pending) return;
  state.pending = requestAnimationFrame(() => { state.pending = null; drawNetwork(); });
}
map.on('zoomend', requestDraw);

// ---- corridors + week grid --------------------------------------------------
function renderCorridors() {
  const t = state.net.slots[state.slot];
  $('corridors').innerHTML = state.cor.corridors.map(c => {
    const i = c.ts.indexOf(t);
    const now = i >= 0 ? c.minutes[i] : null;
    const typ = i >= 0 ? c.typical[i] : null;
    const d = now != null && typ != null ? now - typ : null;
    const cls = d == null ? '' : d > 0.5 ? 'up' : d < -0.5 ? 'down' : '';
    return `<div class="corridor" data-slug="${c.slug}">
      <span class="nm">${c.name}</span>
      <span class="now">${now == null ? '—' : now.toFixed(0)}<small> min</small></span>
      <span class="vs ${cls}">${d == null ? '' : (d > 0 ? '+' : '') + d.toFixed(1)}</span></div>`;
  }).join('');
  document.querySelectorAll('.corridor').forEach(el =>
    el.onclick = () => renderWeek(el.dataset.slug));
}

function renderWeek(slug) {
  const c = state.cor.corridors.find(x => x.slug === slug) || state.cor.corridors[0];
  $('weekName').textContent = '· ' + c.name;
  const days = [], byDay = new Map();
  c.ts.forEach((iso, i) => {
    const d = new Date(iso), key = d.toDateString();
    if (!byDay.has(key)) { byDay.set(key, new Array(24).fill(null)); days.push({ key, d }); }
    const cur = byDay.get(key)[d.getHours()];
    byDay.get(key)[d.getHours()] = cur == null ? c.minutes[i] : Math.max(cur, c.minutes[i]);
  });
  const base = Math.min(...c.minutes);
  let html = '<table class="grid"><tr><th></th>';
  for (let h = 0; h < 24; h++) html += `<th>${h % 3 === 0 ? h : ''}</th>`;
  html += '</tr>';
  for (const { key, d } of days) {
    const wd = d.toLocaleDateString([], { weekday: 'short' });
    html += `<tr><th class="rowhead">${wd}</th>`;
    for (let h = 0; h < 24; h++) {
      const v = byDay.get(key)[h];
      html += `<td style="background:${v == null ? 'transparent' : colourFor(base / v)}"
                   title="${v == null ? '' : `${wd} ${h}:00 · ${v.toFixed(0)} min`}"></td>`;
    }
    html += '</tr>';
  }
  $('week').innerHTML = html + '</table>';
}

function renderFreshness() {
  const el = $('freshness');
  if (!state.cor.generated) return;
  const made = new Date(state.cor.generated);
  const hours = (Date.now() - made) / 3.6e6;
  const when = made.toLocaleString([], { weekday: 'short', hour: 'numeric', minute: '2-digit' });
  // A stalled pipeline would otherwise look identical to a working one.
  el.className = hours > 36 ? 'stale' : '';
  el.textContent = hours > 36
    ? `Forecast last updated ${when}, more than a day ago. It may be out of date. `
    : `Forecast generated ${when}. `;
}

function renderEvents() {
  $('events').innerHTML = state.cor.events.slice(0, 12).map(e => {
    const d = new Date(e.ts);
    return `<li><span class="t">${d.toLocaleDateString([], { weekday: 'short' })} ${fmtHM(d)}</span>
      <span>${e.title || 'Event'}<br><span class="v">${e.venue} · ${e.capacity.toLocaleString()} seats</span></span></li>`;
  }).join('') || '<li>No events inside the forecast horizon.</li>';
}

// ---- place search -----------------------------------------------------------
// The field also accepts a pasted "lat,lon"; the server short-circuits that
// without a network call.
function attachSearch(which) {
  const input = $(which), list = $(which + 'List');
  let timer = null, results = [], active = -1;

  const close = () => { list.hidden = true; active = -1; };
  const choose = i => {
    const r = results[i];
    if (!r) return;
    state.points[which] = [r.lat, r.lon];
    input.value = r.name + (r.address ? `, ${r.address}` : '');
    close();
    dropPin(which);
    if (state.points.from && state.points.to) fitPins();
  };

  input.addEventListener('input', () => {
    state.points[which] = null;
    clearTimeout(timer);
    const q = input.value.trim();
    if (q.length < 3) return close();
    // one request per pause in typing, not one per keystroke
    timer = setTimeout(async () => {
      try {
        const r = await fetch(api('/api/geocode?q=') + encodeURIComponent(q));
        results = (await r.json()).results || [];
      } catch { results = []; }
      if (!results.length) return close();
      list.innerHTML = results.map((x, i) => `
        <li data-i="${i}" class="${x.in_area ? '' : 'outside'}">
          <b>${x.name}</b><span>${x.address || ''}${x.in_area ? '' : ' · outside the forecast area'}</span>
        </li>`).join('');
      list.hidden = false;
      list.querySelectorAll('li').forEach(li =>
        li.onmousedown = e => { e.preventDefault(); choose(+li.dataset.i); });
    }, 250);
  });

  input.addEventListener('keydown', e => {
    if (list.hidden) return;
    const items = list.querySelectorAll('li');
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      active = (active + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
      items.forEach((li, i) => li.classList.toggle('on', i === active));
    } else if (e.key === 'Enter') {
      e.preventDefault(); choose(active >= 0 ? active : 0);
    } else if (e.key === 'Escape') close();
  });
  input.addEventListener('blur', () => setTimeout(close, 120));
}

function dropPin(which) {
  pinLayer.clearLayers();
  for (const k of ['from', 'to']) {
    const p = state.points[k];
    if (!p) continue;
    L.circleMarker(p, {
      radius: 7, color: k === 'from' ? '#16202e' : '#1d5fd0',
      fillColor: '#fff', fillOpacity: 1, weight: 3,
    }).addTo(pinLayer).bindTooltip(k === 'from' ? 'A' : 'B');
  }
}

function fitPins() {
  map.fitBounds(L.latLngBounds([state.points.from, state.points.to]).pad(0.25));
}

map.on('click', e => {
  const which = state.points.from && !state.points.to ? 'to'
    : state.points.to ? 'from' : 'from';
  if (which === 'from') state.points.to = null;
  const ll = [e.latlng.lat, e.latlng.lng];
  state.points[which] = ll;
  $(which).value = `${ll[0].toFixed(4)}, ${ll[1].toFixed(4)}`;
  dropPin(which);
});

// ---- route ------------------------------------------------------------------
function setMode(m) {
  state.mode = m;
  $('mLeave').setAttribute('aria-pressed', m === 'leave');
  $('mArrive').setAttribute('aria-pressed', m === 'arrive');
}
$('mLeave').onclick = () => setMode('leave');
$('mArrive').onclick = () => setMode('arrive');

$('surfaceToggle').onclick = () => {
  state.showSurface = !state.showSurface;
  $('surfaceToggle').setAttribute('aria-pressed', state.showSurface);
  requestDraw();
};

async function forecastRoute() {
  const a = state.points.from, b = state.points.to;
  if (!a || !b) {
    $('answer').classList.add('on');
    $('caveat').textContent = 'Pick a start and an end: search for a place, or click the map.';
    return;
  }
  const when = `${$('day').value}T${$('time').value}:00`;
  const key = state.mode === 'leave' ? 'depart' : 'arrive';
  $('go').disabled = true; $('go').textContent = 'Working…';
  try {
    const url = api(`/api/route?from=${a[0]},${a[1]}&to=${b[0]},${b[1]}&${key}=${when}`);
    const data = await (await fetch(url)).json();
    if (data.error) throw new Error(data.error);
    showRoute(data);
    const p = await (await fetch(api(
      `/api/profile?from=${a[0]},${a[1]}&to=${b[0]},${b[1]}&date=${$('day').value}`))).json();
    drawProfile(p.profile || []);
  } catch (err) {
    $('answer').classList.add('on');
    $('caveat').textContent = 'Could not route: ' + err.message;
  } finally {
    $('go').disabled = false; $('go').textContent = 'Forecast this drive';
  }
}
$('go').onclick = forecastRoute;

function showRoute(data) {
  const s = data.summary;
  $('answer').classList.add('on');
  $('mins').textContent = s.total_minutes.toFixed(0);
  $('arr').textContent = fmtHM(new Date(s.arrive));
  $('answerHead').textContent = state.mode === 'arrive' && data.arrive_by
    ? `Leave at ${fmtHM(new Date(data.arrive_by.depart))}` : 'Forecast';

  const pct = s.total_minutes ? s.freeway_minutes / s.total_minutes * 100 : 0;
  $('barM').style.width = pct + '%';
  $('barE').style.width = (100 - pct) + '%';
  $('keyM').textContent =
    `${s.freeway_minutes.toFixed(0)} min forecast · ${s.freeway_miles.toFixed(1)} mi on ${s.stations_used} detectors`;
  $('keyE').textContent =
    `${s.surface_minutes.toFixed(0)} min inferred · ${s.surface_miles.toFixed(1)} mi`;
  $('caveat').innerHTML = `<b>${Math.round(s.measured_share * 100)}% of this journey is a measured forecast.</b>
    The rest is surface street, inferred by spreading nearby freeway conditions onto local roads,
    the thin lines on the map. That part has no ground truth and is excluded from the published accuracy.`;

  routeLayer.clearLayers();
  const pts = [];
  data.segments.forEach((seg, i) => {
    if (!seg.lat) return;
    pts.push([seg.lat, seg.lon]);
    const prev = data.segments[i - 1];
    if (!prev || !prev.lat) return;
    L.polyline([[prev.lat, prev.lon], [seg.lat, seg.lon]], {
      color: seg.estimate ? '#5a6a80' : '#1d5fd0',
      weight: seg.estimate ? 4 : 7, opacity: 0.95,
      dashArray: seg.estimate ? '3 6' : null,
    }).addTo(routeLayer);
  });
  if (pts.length) map.fitBounds(L.latLngBounds(pts).pad(0.2));

  $('segs').innerHTML =
    '<tr><th>Segment</th><th style="text-align:right">mi</th>' +
    '<th style="text-align:right">mph</th><th style="text-align:right">min</th>' +
    '<th style="text-align:right">at</th></tr>' +
    data.segments.map(x => `<tr class="${x.estimate ? 'est' : ''}">
      <td>${x.estimate ? 'surface streets <i>(inferred)</i>'
                       : `${x.freeway} ${x.direction} · detector ${x.sensor_id}`}</td>
      <td class="n">${x.miles.toFixed(1)}</td><td class="n">${x.mph.toFixed(0)}</td>
      <td class="n">${x.minutes.toFixed(1)}</td>
      <td class="n">${fmtHM(new Date(x.arrive))}</td></tr>`).join('');
}

function drawProfile(rows) {
  const cv = $('profile'), dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = 110;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  if (!rows.length) return;

  const css = getComputedStyle(document.body);
  const ink3 = css.getPropertyValue('--ink-3').trim();
  const accent = css.getPropertyValue('--accent').trim();
  const vals = rows.map(r => r.minutes);
  const lo = Math.min(...vals) * 0.95, hi = Math.max(...vals) * 1.05;
  const x = i => i / (rows.length - 1) * (w - 34) + 30;
  const y = v => h - 18 - (v - lo) / (hi - lo) * (h - 30);

  g.strokeStyle = ink3; g.globalAlpha = 0.25; g.lineWidth = 1;
  [lo, hi].forEach(v => { g.beginPath(); g.moveTo(30, y(v)); g.lineTo(w - 4, y(v)); g.stroke(); });
  g.globalAlpha = 1;

  g.beginPath();
  rows.forEach((r, i) => i ? g.lineTo(x(i), y(r.minutes)) : g.moveTo(x(i), y(r.minutes)));
  g.strokeStyle = accent; g.lineWidth = 2; g.stroke();
  g.lineTo(x(rows.length - 1), h - 18); g.lineTo(x(0), h - 18); g.closePath();
  g.globalAlpha = 0.1; g.fillStyle = accent; g.fill(); g.globalAlpha = 1;

  g.fillStyle = ink3; g.font = '10px ui-monospace, monospace';
  g.fillText(Math.round(hi), 2, y(hi) + 3);
  g.fillText(Math.round(lo), 2, y(lo) + 3);
  [0, 6, 12, 18, 23].forEach(hour => {
    const i = rows.findIndex(r => new Date(r.depart).getHours() === hour);
    if (i >= 0) g.fillText(hour + ':00', x(i) - 10, h - 5);
  });
  const worst = vals.indexOf(Math.max(...vals));
  g.fillStyle = accent;
  g.beginPath(); g.arc(x(worst), y(vals[worst]), 3, 0, 7); g.fill();
}

// ---- boot -------------------------------------------------------------------
(async function () {
  drawLegend();
  API = await fetch('config.json').then(r => r.json()).then(c => c.api_base || '')
                                  .catch(() => '');
  const [net, cor, geo] = await Promise.all([
    fetch('data/network.json').then(r => r.json()),
    fetch('data/corridors.json').then(r => r.json()),
    fetch('data/geometry.json').then(r => r.json()),
  ]);
  Object.assign(state, { net, cor, geo });

  $('day').value = net.slots[0].slice(0, 10);
  $('day').min = net.slots[0].slice(0, 10);
  $('day').max = net.slots[net.slots.length - 1].slice(0, 10);
  $('slot').max = net.slots.length - 1;

  // open on the next weekday morning peak, the question the site exists to
  // answer, rather than midnight, the least interesting hour of the week
  const peak = net.slots.findIndex(s => {
    const d = new Date(s);
    return d.getHours() === 8 && d.getDay() >= 1 && d.getDay() <= 5;
  });
  state.slot = peak >= 0 ? peak : 8;
  $('slot').value = state.slot;
  $('slot').oninput = e => { state.slot = +e.target.value; requestDraw(); };

  ['from', 'to'].forEach(attachSearch);
  drawNetwork();
  renderWeek(cor.corridors[0].slug);
  renderFreshness();
  renderEvents();
})();
