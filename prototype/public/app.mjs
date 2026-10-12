const $ = id => document.getElementById(id);
let token, state, keyPair, busy = false;
let noticeTimer;
function notice(text) {
  $('status').textContent = text;
  $('status').classList.add('visible');
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => $('status').classList.remove('visible'), 7000);
}
async function api(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Demo-Token': token },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error ?? 'Request failed');
  return data;
}
function actionText(action) {
  if (action.type === 'general_information') return 'General information only — no account change';
  return `Increase transfer limit from ₹${action.from.toLocaleString('en-IN')} to ₹${action.to.toLocaleString('en-IN')}?`;
}
function render() {
  $('start').disabled = busy;
  for (const id of ['observe', 'challenge', 'approve', 'reject', 'release', 'callback', 'cancel']) $(id).disabled = true;
  if (!state) return;
  $('scenarioName').textContent = state.name;
  $('action').textContent = actionText(state.action);
  $('voice').textContent = state.voice.replaceAll('_', ' ');
  $('confirmation').textContent = state.confirmation.replaceAll('_', ' ');
  $('outcome').textContent = state.outcome.replaceAll('_', ' ');
  $('message').textContent = state.message;
  $('windows').replaceChildren(...state.observations.map(o => {
    const el = document.createElement('div'); el.className = 'window';
    el.textContent = `Fixture ${o.sequence}: ${o.score === null ? 'UNKNOWN' : o.score.toFixed(2)}`;
    return el;
  }));
  $('events').replaceChildren(...state.events.map(e => {
    const li = document.createElement('li');
    const type = document.createElement('span'); type.className = 'event-type'; type.textContent = e.type;
    li.append(type, document.createTextNode(` · ${new Date(e.at).toLocaleTimeString()} — ${e.detail}`));
    return li;
  }));
  const closed = state.outcome === 'request_cancelled';
  const pending = state.confirmation === 'pending';
  $('observe').disabled = busy || closed || state.observations.length >= 3;
  $('challenge').disabled = busy || closed || !state.sensitive || pending || state.confirmation === 'approved';
  $('approve').disabled = $('reject').disabled = busy || closed || !pending || !keyPair;
  $('release').disabled = busy || closed || (state.sensitive && state.confirmation !== 'approved');
  $('callback').disabled = $('cancel').disabled = busy || closed;
  $('customerAction').textContent = pending ? actionText(state.action) : 'No pending confirmation';
  $('expiry').textContent = pending ? `Challenge expires at ${new Date(state.challenge.expires_at).toLocaleTimeString()}; one-use, exact-action bound.` : '';
  $('keyState').textContent = state.registered ? 'Real P-256 browser signature available · DEMO enrollment only' : 'No demo key registered';
}
async function run(fn) {
  if (busy) return;
  busy = true; render();
  try { await fn(); }
  catch (error) { notice(error.message); }
  finally { busy = false; render(); }
}
async function op(name, body = {}) {
  state = await api(`/api/session/${state.id}/${name}`, body);
}
$('start').addEventListener('click', () => run(async () => {
  state = await api('/api/session', { scenario: $('scenario').value });
  keyPair = await crypto.subtle.generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, false, ['sign', 'verify']);
  const jwk = await crypto.subtle.exportKey('jwk', keyPair.publicKey);
  await op('credential', { jwk });
  notice('New demo session. Acoustic windows are simulated; no audio is analysed.');
}));
$('observe').addEventListener('click', () => run(() => op('observe')));
$('challenge').addEventListener('click', () => run(() => op('challenge')));
function message(c, decision) {
  return JSON.stringify([c.domain, c.challenge_id, c.customer_id, c.session_id,
    c.credential_id, c.action_version, c.action_digest, c.nonce, c.expires_at, decision]);
}
function base64url(bytes) {
  return btoa(String.fromCharCode(...new Uint8Array(bytes))).replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '');
}
async function respond(decision) {
  const c = state.challenge;
  const signature = await crypto.subtle.sign({ name: 'ECDSA', hash: 'SHA-256' }, keyPair.privateKey,
    new TextEncoder().encode(message(c, decision)));
  await op('respond', { challenge_id: c.challenge_id, decision, signature: base64url(signature) });
}
$('approve').addEventListener('click', () => run(() => respond('approve')));
$('reject').addEventListener('click', () => run(() => respond('reject')));
for (const decision of ['release', 'callback', 'cancel']) {
  $(decision).addEventListener('click', () => run(() => op('analyst', { decision, reason: $('reason').value })));
}
// Poll only for expiry/status changes; no media or device/call-state collection.
setInterval(async () => {
  if (busy || !state || state.confirmation !== 'pending') return;
  try { state = await api(`/api/session/${state.id}`); render(); } catch { /* local server may have stopped */ }
}, 1000);
try {
  const bootstrap = await api('/api/bootstrap'); token = bootstrap.token;
  for (const scenario of bootstrap.scenarios) {
    const option = document.createElement('option'); option.value = scenario.id; option.textContent = scenario.name;
    $('scenario').append(option);
  }
  render();
} catch (error) { notice(error.message); $('start').disabled = true; }
