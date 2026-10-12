import { createServer } from 'node:http';
import { randomBytes } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { Workflow, WorkflowError, SCENARIOS } from './core.mjs';

export function makeServer() {
  const workflow = new Workflow();
  // CSRF guard for the local demo, NOT customer/analyst authentication.
  const token = randomBytes(32).toString('base64url');
  const staticFiles = new Map([
    ['/', ['public/index.html', 'text/html; charset=utf-8']],
    ['/app.mjs', ['public/app.mjs', 'text/javascript; charset=utf-8']],
    ['/style.css', ['public/style.css', 'text/css; charset=utf-8']],
  ]);
  const json = (res, status, data) => {
    res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify(data));
  };
  const readBody = async req => {
    let size = 0; const chunks = [];
    for await (const chunk of req) {
      size += chunk.length;
      if (size > 16_384) throw new WorkflowError('Request too large', 413);
      chunks.push(chunk);
    }
    try { return JSON.parse(Buffer.concat(chunks).toString('utf8')); }
    catch { throw new WorkflowError('Invalid JSON'); }
  };
  const server = createServer(async (req, res) => {
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Referrer-Policy', 'no-referrer');
    res.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'");
    try {
      // Restrict Host to loopback to reduce DNS-rebinding risk. Server also binds loopback.
      if (!/^(localhost|127\.0\.0\.1)(:\d+)?$/.test(req.headers.host ?? '')) {
        throw new WorkflowError('Local demo only', 403);
      }
      const url = new URL(req.url, 'http://' + req.headers.host);
      if (req.headers.origin && req.headers.origin !== 'http://' + req.headers.host) {
        throw new WorkflowError('Cross-origin request rejected', 403);
      }
      if (req.method === 'GET' && staticFiles.has(url.pathname)) {
        const [path, type] = staticFiles.get(url.pathname);
        const data = await readFile(new URL(path, import.meta.url));
        res.writeHead(200, { 'Content-Type': type }); res.end(data); return;
      }
      if (req.method === 'GET' && url.pathname === '/api/bootstrap') {
        json(res, 200, { token, scenarios: Object.entries(SCENARIOS).map(([id, s]) => ({ id, name: s.name })),
          disclaimer: 'Local workflow demo. All acoustic scores are simulated. No bank authentication, biometric verification or real action execution.' }); return;
      }
      if (req.headers['x-demo-token'] !== token) throw new WorkflowError('Local demo token required', 403);
      if (req.method === 'POST' && url.pathname === '/api/session') {
        const body = await readBody(req);
        json(res, 201, workflow.create(body.scenario)); return;
      }
      const match = /^\/api\/session\/([0-9a-f-]+)(?:\/(credential|observe|challenge|respond|analyst))?$/.exec(url.pathname);
      if (!match) throw new WorkflowError('Not found', 404);
      const [, id, operation] = match;
      if (req.method === 'GET' && !operation) { json(res, 200, workflow.view(id)); return; }
      if (req.method !== 'POST' || !operation) throw new WorkflowError('Method not supported', 405);
      if (!req.headers['content-type']?.startsWith('application/json')) throw new WorkflowError('JSON body required', 415);
      const body = await readBody(req);
      let state;
      if (operation === 'credential') state = workflow.registerCredential(id, body.jwk);
      if (operation === 'observe') state = workflow.nextObservation(id);
      if (operation === 'challenge') state = workflow.issueChallenge(id);
      if (operation === 'respond') state = workflow.respond(id, body);
      if (operation === 'analyst') state = workflow.analystDecision(id, body);
      json(res, 200, state);
    } catch (error) {
      json(res, error instanceof WorkflowError ? error.status : 500,
        { error: error instanceof WorkflowError ? error.message : 'Unexpected local server error' });
    }
  });
  server.requestTimeout = 15_000;
  server.headersTimeout = 10_000;
  return server;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const port = Number(process.env.PORT ?? 8787);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid PORT');
  makeServer().listen(port, '127.0.0.1', () => {
    console.log(`EchoLock workflow demo: http://localhost:${port}`);
    console.log('SIMULATION ONLY: no audio detector, no bank integration, no biometric attestation.');
  });
}
