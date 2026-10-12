import test from 'node:test';
import assert from 'node:assert/strict';
import { request } from 'node:http';
import { makeServer } from '../server.mjs';

test('HTTP app: static assets, simulation API, token and origin checks', async t => {
  const server = makeServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  const base = `http://127.0.0.1:${server.address().port}`;
  const page = await fetch(base); assert.equal(page.status, 200);
  assert.match(await page.text(), /SIMULATION, NOT A VOICE DETECTOR/);
  for (const file of ['/app.mjs', '/style.css']) assert.equal((await fetch(base + file)).status, 200);
  const bootstrap = await (await fetch(base + '/api/bootstrap')).json();
  const withoutToken = await fetch(base + '/api/session', { method: 'POST', body: '{}' });
  assert.equal(withoutToken.status, 403);
  const headers = { 'Content-Type': 'application/json', 'X-Demo-Token': bootstrap.token };
  const badOrigin = await fetch(base + '/api/session', { method: 'POST', headers: { ...headers, Origin: 'https://attacker.invalid' }, body: '{"scenario":"clone"}' });
  assert.equal(badOrigin.status, 403);
  const response = await fetch(base + '/api/session', { method: 'POST', headers, body: '{"scenario":"clone"}' });
  assert.equal(response.status, 201); const state = await response.json();
  const observation = await fetch(`${base}/api/session/${state.id}/observe`, { method: 'POST', headers, body: '{}' });
  assert.equal(observation.status, 200);
  assert.equal((await observation.json()).observations[0].provenance, 'simulated_fixture');
  const invalid = await fetch(base + '/api/session', { method: 'POST', headers, body: 'not JSON' });
  assert.equal(invalid.status, 400);
  const oversized = await fetch(base + '/api/session', { method: 'POST', headers, body: JSON.stringify({ padding: 'x'.repeat(20_000) }) });
  assert.equal(oversized.status, 413);
  // Fetch normalises Host; use the raw HTTP client for DNS-rebinding coverage.
  const badHostStatus = await new Promise((resolve, reject) => {
    const req = request(base + '/api/bootstrap', { headers: { Host: 'evil.invalid' } }, res => {
      res.resume(); res.on('end', () => resolve(res.statusCode));
    });
    req.on('error', reject); req.end();
  });
  assert.equal(badHostStatus, 403);
});
