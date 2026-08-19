// site/app.js
// The map is a rendering of one number per detector per hour, so nearly all of
// this file is about making that number legible: a speed scale that reads at a
// glance, a scrubber that moves through the week, and a route panel that keeps
// measured forecast and estimated surface time visually distinct at every step.

const SCALE = [            // ratio of forecast speed to that detector's free-flow
  [0.95, '#1f9d84', 'free'],
  [0.85, '#7cb342', ''],
  [0.70, '#e8a020', ''],
  [0.50, '#dd6a1f', ''],
  [0.00, '#c0392b', 'jam'],
];
const colourFor = r => (SCALE.find(s => r >= s[0]) || SCALE[SCALE.length - 1])[1];

const $ = id => document.getElementById(id);
const fmtHM = d => d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
const state = { net: null, cor: null, slot: 8, layers: [], route: null, mode: 'leave' };

// ---- map --------------------------------------------------------------------
const map = L.map('map', { zoomControl: true }).setView([37.72, -122.15], 10);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap, &copy; CARTO', maxZoom: 18,
}).addTo(map);
const routeLayer = L.layerGroup().addTo(map);
const pinLayer = L.layerGroup().addTo(map);

function drawLegend() {
  $('legend').innerHTML = '<span>jam</span>' +
    SCALE.slice().reverse().map(s => `<i style="background:${s[1]}"></i>`).join('') +
    '<span>free</span>';
}

function drawNetwork() {
  const { segments, speeds, freeflow } = state.net;
  state.layers.forEach(l => map.removeLayer(l));
  state.layers = [];
  const bySpeed = new Map();
  for (const seg of segments) {
    const v = speeds[seg.id]?.[state.slot];
    if (v == null || v < 0) continue;
    const ff = freeflow[seg.id] || 65;
    const c = colourFor(v / ff);
    if (!bySpeed.has(c)) bySpeed.set(c, []);
    bySpeed.get(c).push([seg.a, seg.b]);
  }
  // one polyline layer per colour instead of 1,900 individual layers: same
  // picture, a fraction of the DOM and it redraws instantly when scrubbing
  for (const [colour, lines] of bySpeed) {
    const layer = L.polyline(lines, { color: colour, weight: 3, opacity: 0.9 });
    layer.addTo(map);
    state.layers.push(layer);
  }
  const t = new Date(state.net.slots[state.slot]);
  $('slotLabel').textContent = t.toLocaleDateString([], { weekday: 'long' }) + ' ' + fmtHM(t);
  renderCorridors();
}

// ---- corridors + week grid --------------------------------------------------
function renderCorridors() {
  const t = state.net.slots[state.slot];
  const rows = state.cor.corridors.map(c => {
    const i = c.ts.indexOf(t);
    const now = i >= 0 ? c.minutes[i] : null;
    const typ = i >= 0 ? c.typical[i] : null;
    const d = now != null && typ != null ? now - typ : null;
    const cls = d == null ? '' : d > 0.5 ? 'up' : d < -0.5 ? 'down' : '';
    const sign = d == null ? '' : (d > 0 ? '+' : '');
    return `<div class="corridor" data-slug="${c.slug}">
      <span class="nm">${c.name}</span>
      <span class="now">${now == null ? '—' : now.toFixed(0)}<small> min</small></span>
      <span class="vs ${cls}">${d == null ? '' : sign + d.toFixed(1)}</span></div>`;
  });
  $('corridors').innerHTML = rows.join('');
  document.querySelectorAll('.corridor').forEach(el =>
    el.onclick = () => renderWeek(el.dataset.slug));
}

function renderWeek(slug) {
  const c = state.cor.corridors.find(x => x.slug === slug) || state.cor.corridors[0];
  $('weekName').textContent = '· ' + c.name;
  const days = [], byDay = new Map();
  c.ts.forEach((iso, i) => {
    const d = new Date(iso);
    const key = d.toDateString();
    if (!byDay.has(key)) { byDay.set(key, new Array(24).fill(null)); days.push({ key, d }); }
    // several 15-minute slots share an hour; keep the worst, since that is what
    // a commuter planning around the hour actually cares about
    const cur = byDay.get(key)[d.getHours()];
    byDay.get(key)[d.getHours()] = cur == null ? c.minutes[i] : Math.max(cur, c.minutes[i]);
  });
  const base = Math.min(...c.minutes);
  let html = '<table class="grid"><tr><th></th>';
  for (let h = 0; h < 24; h++) html += `<th>${h % 3 === 0 ? h : ''}</th>`;
  html += '</tr>';
  for (const { key, d } of days) {
    html += `<tr><th class="rowhead">${d.toLocaleDateString([], { weekday: 'short' })}</th>`;
    for (let h = 0; h < 24; h++) {
      const v = byDay.get(key)[h];
      const colour = v == null ? 'transparent' : colourFor(base / v);
      const label = v == null ? '' : `${d.toLocaleDateString([], { weekday: 'short' })} ${h}:00 — ${v.toFixed(0)} min`;
      html += `<td style="background:${colour}" title="${label}"></td>`;
    }
    html += '</tr>';
  }
  $('week').innerHTML = html + '</table>';
}

function renderEvents() {
  $('events').innerHTML = state.cor.events.slice(0, 12).map(e => {
    const d = new Date(e.ts);
    return `<li><span class="t">${d.toLocaleDateString([], { weekday: 'short' })} ${fmtHM(d)}</span>
      <span>${e.title || 'Event'}<br><span class="v">${e.venue} · ${e.capacity.toLocaleString()} seats</span></span></li>`;
  }).join('') || '<li>No events inside the forecast horizon.</li>';
}

// ---- route ------------------------------------------------------------------
let pins = [];
map.on('click', e => {
  const ll = `${e.latlng.lat.toFixed(4)},${e.latlng.lng.toFixed(4)}`;
  if (pins.length >= 2) { pins = []; pinLayer.clearLayers(); }
  pins.push(ll);
  $(pins.length === 1 ? 'from' : 'to').value = ll;
  L.circleMarker(e.latlng, {
    radius: 7, color: pins.length === 1 ? '#16202e' : '#1d5fd0',
    fillColor: '#fff', fillOpacity: 1, weight: 3,
  }).addTo(pinLayer);
});

function setMode(m) {
  state.mode = m;
  $('mLeave').setAttribute('aria-pressed', m === 'leave');
  $('mArrive').setAttribute('aria-pressed', m === 'arrive');
}
$('mLeave').onclick = () => setMode('leave');
$('mArrive').onclick = () => setMode('arrive');

async function forecastRoute() {
  const from = $('from').value.trim(), to = $('to').value.trim();
  const when = `${$('day').value}T${$('time').value}:00`;
  const key = state.mode === 'leave' ? 'depart' : 'arrive';
  $('go').disabled = true; $('go').textContent = 'Working…';
  try {
    const r = await fetch(`/api/route?from=${from}&to=${to}&${key}=${when}`);
    const data = await r.json();
    if (data.error) throw new Error(data.error);
    showRoute(data);
    loadProfile(from, to, $('day').value);
  } catch (err) {
    $('caveat').textContent = 'Could not route: ' + err.message;
    $('answer').classList.add('on');
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
  $('keyM').textContent = `${s.freeway_minutes.toFixed(0)} min forecast · ${s.freeway_miles.toFixed(1)} mi on ${s.stations_used} detectors`;
  $('keyE').textContent = `${s.surface_minutes.toFixed(0)} min estimated · ${s.surface_miles.toFixed(1)} mi`;
  $('caveat').innerHTML = `<b>${Math.round(s.measured_share * 100)}% of this journey is a measured forecast.</b>
    The rest is surface street, estimated by spreading nearby freeway conditions onto local roads.
    That part has no ground truth and is excluded from the published accuracy figures.`;

  routeLayer.clearLayers();
  const pts = data.segments.filter(x => x.lat).map(x => [x.lat, x.lon]);
  data.segments.forEach((seg, i) => {
    if (i === 0 || !seg.lat) return;
    const prev = data.segments[i - 1];
    if (!prev.lat) return;
    L.polyline([[prev.lat, prev.lon], [seg.lat, seg.lon]], {
      color: seg.estimate ? '#8493a8' : '#1d5fd0',
      weight: seg.estimate ? 4 : 6, opacity: 0.95,
      dashArray: seg.estimate ? '4 5' : null,
    }).addTo(routeLayer);
  });
  if (pts.length) map.fitBounds(L.latLngBounds(pts).pad(0.2));

  $('segs').innerHTML = '<tr><th>Segment</th><th style="text-align:right">mi</th>' +
    '<th style="text-align:right">mph</th><th style="text-align:right">min</th>' +
    '<th style="text-align:right">at</th></tr>' +
    data.segments.map(x => `<tr class="${x.estimate ? 'est' : ''}">
      <td>${x.estimate ? 'surface streets' : 'I-' + x.freeway + ' ' + x.direction + ' · ' + x.sensor_id}</td>
      <td class="n">${x.miles.toFixed(1)}</td><td class="n">${x.mph.toFixed(0)}</td>
      <td class="n">${x.minutes.toFixed(1)}</td>
      <td class="n">${fmtHM(new Date(x.arrive))}</td></tr>`).join('');
}

async function loadProfile(from, to, date) {
  const r = await fetch(`/api/profile?from=${from}&to=${to}&date=${date}`);
  const { profile } = await r.json();
  drawProfile(profile);
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

  g.beginPath(); rows.forEach((r, i) => i ? g.lineTo(x(i), y(r.minutes)) : g.moveTo(x(i), y(r.minutes)));
  g.strokeStyle = accent; g.lineWidth = 2; g.stroke();
  g.lineTo(x(rows.length - 1), h - 18); g.lineTo(x(0), h - 18); g.closePath();
  g.globalAlpha = 0.10; g.fillStyle = accent; g.fill(); g.globalAlpha = 1;

  g.fillStyle = ink3; g.font = '10px ui-monospace, monospace';
  g.fillText(Math.round(hi) + '', 2, y(hi) + 3);
  g.fillText(Math.round(lo) + '', 2, y(lo) + 3);
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
  const [net, cor] = await Promise.all([
    fetch('data/network.json').then(r => r.json()),
    fetch('data/corridors.json').then(r => r.json()),
  ]);
  state.net = net; state.cor = cor;
  $('day').value = net.slots[0].slice(0, 10);
  $('day').min = net.slots[0].slice(0, 10);
  $('day').max = net.slots[net.slots.length - 1].slice(0, 10);
  $('slot').max = net.slots.length - 1;

  // open on the next weekday morning peak, which is the question the site exists
  // to answer -- not midnight, which is the least interesting hour of the week
  const peak = net.slots.findIndex(s => {
    const d = new Date(s);
    return d.getHours() === 8 && d.getDay() >= 1 && d.getDay() <= 5;
  });
  state.slot = peak >= 0 ? peak : 8;
  $('slot').value = state.slot;
  $('slot').oninput = e => { state.slot = +e.target.value; drawNetwork(); };

  drawNetwork();
  renderWeek(cor.corridors[0].slug);
  renderEvents();
})();
