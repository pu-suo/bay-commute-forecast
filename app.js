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
let apiUp = null;          // null until probed
const api = path => API + path;

// A missing API answers with the host's own 404 page, so res.json() reports a
// stray "<" rather than anything a visitor can act on. Check the response
// before trusting it is JSON.
async function getJSON(url) {
  let res;
  try {
    res = await fetch(url);
  } catch {
    throw new Error('the forecast service is unreachable');
  }
  const type = res.headers.get('content-type') || '';
  if (!type.includes('application/json')) {
    throw new Error(res.status === 404
      ? 'the forecast service is not deployed at this address'
      : `the forecast service returned ${res.status}`);
  }
  const body = await res.json();
  if (res.status === 429) throw new Error('too many requests just now, try again shortly');
  if (!res.ok || body.error) throw new Error(body.error || `request failed (${res.status})`);
  return body;
}

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
// Esri's gray canvas, which needs no key. CARTO's free tiles now come back
// stamped API KEY REQUIRED. Base and labels are separate layers here; both sit
// in the tile pane, so the road colours draw over them as before.
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas';
const basemapAttr =
  'Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors';
// The canvas stops at z16; let Leaflet upscale rather than show blank tiles.
L.tileLayer(`${ESRI}/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}`, {
  attribution: basemapAttr, maxZoom: 18, maxNativeZoom: 16,
}).addTo(map);
L.tileLayer(`${ESRI}/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}`, {
  maxZoom: 18, maxNativeZoom: 16,
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
    el.onclick = () => {
      renderWeek(el.dataset.slug);
      // Picking a corridor is a request to see its week, so open that section
      // even if it was collapsed, and scroll it into view.
      const fold = $('foldWeek');
      if (!fold.open) fold.open = true;
      fold.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    });
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

// Remember which sections the visitor left open. Cheap, and the alternative is
// re-collapsing the thing they were reading every time the page reloads.
function restoreFolds() {
  document.querySelectorAll('details.fold').forEach(d => {
    const saved = localStorage.getItem('fold:' + d.id);
    if (saved !== null) d.open = saved === '1';
    d.addEventListener('toggle', () =>
      localStorage.setItem('fold:' + d.id, d.open ? '1' : '0'));
  });
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
        results = (await getJSON(api('/api/geocode?q=') + encodeURIComponent(q))).results || [];
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

// Long enough to cover a container waking from idle, short enough that a
// genuinely absent service is reported rather than spun on forever.
const PROBE_TIMEOUT_MS = 12000;

async function probeApi() {
  try {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), PROBE_TIMEOUT_MS);
    try {
      const res = await fetch(api('/api/health'), { signal: ctl.signal });
      if (!(res.headers.get('content-type') || '').includes('application/json')) {
        throw new Error('not the forecast service');
      }
      await res.json();
    } finally {
      clearTimeout(timer);
    }
    apiUp = true;
  } catch (err) {
    apiUp = false;
    $('go').disabled = true;
    $('go').textContent = 'Route planning unavailable';
    $('planNote').textContent =
      'Route planning needs the forecast service, which is not reachable from here. '
      + 'The map, the week ahead and the accuracy page all work without it.';
    $('planNote').hidden = false;
    ['from', 'to'].forEach(id => { $(id).disabled = true; });
  }
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
    const data = await getJSON(
      api(`/api/route?from=${a[0]},${a[1]}&to=${b[0]},${b[1]}&${key}=${when}`));
    showRoute(data);
    const p = await getJSON(api(
      `/api/profile?from=${a[0]},${a[1]}&to=${b[0]},${b[1]}&date=${$('day').value}`));
    drawProfile(p.profile || []);
  } catch (err) {
    $('answer').classList.add('on');
    $('mins').textContent = '--';
    $('arr').textContent = '--';
    $('caveat').textContent = 'Could not plan this drive: ' + err.message + '.';
  } finally {
    $('go').disabled = false; $('go').textContent = 'Forecast this drive';
  }
}
$('go').onclick = forecastRoute;

// ---- keeping the map and the plan on the same clock -------------------------
// The map holds one frame per hour: build_data.py keeps only the minute==0 rows
// of a forecast that is really 15-minute. The planner asks in 15-minute steps.
// So the two can only ever meet on the hour, and the rounding has to land
// somewhere.
//
// It lands on the map. Dragging the slider writes the form exactly, because a
// slot is always a whole hour and the time input holds that without loss. A
// planned drive only nudges the slider, and never rewrites the time the user
// typed: rounding 8:45 down to 8:00 in the form would quietly answer a
// different question from the one still on screen.

// The API returns "2026-08-28 08:05:20.112290435" -- a space where ISO wants a
// T, and more fractional digits than Date parses on every engine.
function parseLocal(v) {
  if (!v) return NaN;
  return new Date(String(v).replace(' ', 'T').replace(/(\.\d{3})\d+$/, '$1')).getTime();
}

function nearestSlot(when) {
  const t = parseLocal(when);
  if (!state.net || !Number.isFinite(t)) return -1;
  let best = -1, gap = Infinity;
  state.net.slots.forEach((iso, i) => {
    const d = Math.abs(parseLocal(iso) - t);
    if (d < gap) { gap = d; best = i; }
  });
  return best;
}

// Follows departure, not arrival: departure is when the driving starts, and
// under "arrive by" the server is the only thing that knows it. A drive spans
// slots anyway -- 8:20 to 8:50 straddles two -- so one frame is always a
// simplification, and the departure frame is the honest one to show.
function syncMapToDrive(depart) {
  const i = nearestSlot(depart);
  if (i < 0) return '';
  if (i !== state.slot) {
    state.slot = i;
    $('slot').value = i;
    requestDraw();
  }
  const slotAt = new Date(parseLocal(state.net.slots[i]));
  const driveAt = new Date(parseLocal(depart));
  const label = `${slotAt.toLocaleDateString([], { weekday: 'long' })} ${fmtHM(slotAt)}`;
  return Math.abs(slotAt - driveAt) < 60000
    ? ` The map is showing ${label}, the hour this drive starts in.`
    : ` The map is showing ${label}, the closest hour it has to leaving at ${fmtHM(driveAt)}.`;
}

// Lossless in this direction: a slot is a whole hour, which step="900" holds.
function syncPlanToMap() {
  const iso = state.net && state.net.slots[state.slot];
  if (!iso) return;
  $('day').value = iso.slice(0, 10);
  $('time').value = iso.slice(11, 16);
}

// ---- events on this drive ---------------------------------------------------
// These constants are the model's, not the page's: features_stations.attach_events
// attaches a venue to a detector within 12 miles, and counts an event live from
// 4 hours before the doors to 6 hours after. Saying anything wider would claim
// the forecast accounts for something it does not. If those numbers move in the
// trainer, move them here too.
const EVENT_MILES = 12, EVENT_BEFORE_H = 4, EVENT_AFTER_H = 6;

function milesApart(lat1, lon1, lat2, lon2) {
  // equirectangular, same approximation the feature builder uses
  const dy = (lat1 - lat2) * 69.0;
  const dx = (lon1 - lon2) * 69.0 * Math.cos(lat1 * Math.PI / 180);
  return Math.sqrt(dx * dx + dy * dy);
}

function eventsOnDrive(data, departMs, arriveMs) {
  const evs = (state.cor && state.cor.events) || [];
  const pts = data.segments.filter(s => s.lat);
  if (!pts.length || !evs.length) return [];
  return evs.map(e => {
    const start = parseLocal(e.ts);
    if (!Number.isFinite(start) || e.lat == null) return null;
    if (arriveMs < start - EVENT_BEFORE_H * 3.6e6) return null;
    if (departMs > start + EVENT_AFTER_H * 3.6e6) return null;
    let near = Infinity;
    for (const p of pts) {
      near = Math.min(near, milesApart(p.lat, p.lon, e.lat, e.lon));
      if (near <= EVENT_MILES) break;
    }
    return near <= EVENT_MILES ? { ...e, start, near } : null;
  }).filter(Boolean).sort((a, b) => a.near - b.near).slice(0, 2);
}

function renderDriveEvents(data, departMs, arriveMs) {
  const hits = eventsOnDrive(data, departMs, arriveMs);
  const box = $('evnote');
  if (!hits.length) { box.hidden = true; box.innerHTML = ''; return; }
  box.innerHTML = hits.map(e => {
    const t = new Date(e.start);
    const when = `${t.toLocaleDateString([], { weekday: 'short' })} ${fmtHM(t)}`;
    return `<b>${e.title}</b> at ${e.venue}, ${when} — ${e.capacity.toLocaleString()} seats,
            ${e.near.toFixed(0)} mi from this route. The forecast already accounts for it.`;
  }).join('<br>');
  box.hidden = false;
}

function showRoute(data) {
  const s = data.summary;
  $('answer').classList.add('on');
  $('mins').textContent = s.total_minutes.toFixed(0);
  $('arr').textContent = fmtHM(new Date(s.arrive));
  $('answerHead').textContent = state.mode === 'arrive' && data.arrive_by
    ? `Leave at ${fmtHM(new Date(data.arrive_by.depart))}` : 'Forecast';

  // The spread of days behind the single number. Not an error bar on the model:
  // it is how much this drive varies between matching weekdays, which is the
  // question a lone figure quietly answers "not at all".
  const band = $('band');
  if (s.typical_fast != null && s.typical_slow != null) {
    const day = new Date(parseLocal(s.depart))
      .toLocaleDateString([], { weekday: 'long' });
    band.innerHTML = `usually <b>${Math.round(s.typical_fast)}\u2013${Math.round(s.typical_slow)} min</b> on a ${day}`;
    band.hidden = false;
  } else {
    band.hidden = true; band.innerHTML = '';
  }

  const pct = s.total_minutes ? s.freeway_minutes / s.total_minutes * 100 : 0;
  $('barM').style.width = pct + '%';
  $('barE').style.width = (100 - pct) + '%';
  $('keyM').textContent =
    `${s.freeway_minutes.toFixed(0)} min forecast · ${s.freeway_miles.toFixed(1)} mi on ${s.stations_used} detectors`;
  $('keyE').textContent =
    `${s.surface_minutes.toFixed(0)} min inferred · ${s.surface_miles.toFixed(1)} mi`;
  const departAt = state.mode === 'arrive' && data.arrive_by
    ? data.arrive_by.depart : s.depart;
  const mapNote = syncMapToDrive(departAt);
  renderDriveEvents(data, parseLocal(departAt), parseLocal(s.arrive));
  $('caveat').innerHTML = `<b>${Math.round(s.measured_share * 100)}% of this journey is a measured forecast.</b>
    The rest is surface street, inferred by spreading nearby freeway conditions onto local roads,
    the thin lines on the map. That part has no ground truth and is excluded from the published accuracy.`
    + mapNote;

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
  // All four in parallel. config.json used to be awaited first, which cost a
  // whole round trip before the data even started.
  const [cfg, net, cor, geo] = await Promise.all([
    fetch('config.json').then(r => r.json()).catch(() => ({})),
    fetch('data/network.json').then(r => r.json()),
    fetch('data/corridors.json').then(r => r.json()),
    fetch('data/geometry.json').then(r => r.json()),
  ]);
  API = cfg.api_base || '';
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
  syncPlanToMap();
  $('slot').oninput = e => {
    state.slot = +e.target.value;
    syncPlanToMap();
    requestDraw();
  };

  ['from', 'to'].forEach(attachSearch);
  restoreFolds();
  drawNetwork();
  renderWeek(cor.corridors[0].slug);
  renderFreshness();
  renderEvents();
  // Deliberately not awaited. The map is the page; it must not wait on a
  // service it does not need. A sleeping container can take seconds to wake,
  // and blocking here turned that into a blank screen.
  probeApi();
})();
