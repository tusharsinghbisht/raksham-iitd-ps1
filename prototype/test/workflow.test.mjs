import test from 'node:test';
import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';
import { Workflow, confirmationMessage, actionDigest } from '../core.mjs';

async function setup(scenario = 'genuine', options = {}) {
  const w = new Workflow(options);
  const s = w.create(scenario);
  const keys = await webcrypto.subtle.generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, false, ['sign', 'verify']);
  const jwk = await webcrypto.subtle.exportKey('jwk', keys.publicKey);
  w.registerCredential(s.id, jwk);
  return { w, id: s.id, keys };
}
async function signature(keys, challenge, decision = 'approve') {
  const bytes = await webcrypto.subtle.sign({ name: 'ECDSA', hash: 'SHA-256' }, keys.privateKey,
    new TextEncoder().encode(confirmationMessage(challenge, decision)));
  return Buffer.from(bytes).toString('base64url');
}
async function approve(f) {
  const c = f.w.issueChallenge(f.id).challenge;
  const sig = await signature(f.keys, c);
  return f.w.respond(f.id, { challenge_id: c.challenge_id, decision: 'approve', signature: sig });
}
function three(f) { for (let i = 0; i < 3; i++) f.w.nextObservation(f.id); }

test('sensitive action requires confirmation even before any acoustic observation', () => {
  const w = new Workflow(); const s = w.create('genuine');
  assert.equal(s.voice, 'unknown'); assert.equal(s.outcome, 'confirmation_required');
});
test('genuine fixture becomes not elevated but still cannot authorise an action', async () => {
  const f = await setup(); three(f);
  const s = f.w.view(f.id); assert.equal(s.voice, 'not_elevated');
  assert.equal(s.outcome, 'confirmation_required');
});
test('three high fixtures latch elevated suspicion; one does not decide', async () => {
  const f = await setup('clone'); f.w.nextObservation(f.id);
  assert.equal(f.w.view(f.id).voice, 'unknown');
  f.w.nextObservation(f.id); assert.equal(f.w.view(f.id).voice, 'unknown');
  f.w.nextObservation(f.id);
  assert.equal(f.w.view(f.id).outcome, 'confirmation_and_review');
});
test('unknown audio never becomes genuine and still uses secure confirmation', async () => {
  const f = await setup('unknown'); three(f);
  assert.equal(f.w.view(f.id).voice, 'unknown');
  assert.equal((await approve(f)).outcome, 'eligible_for_bank_checks');
});
test('a missed clone still requires action confirmation', async () => {
  const f = await setup('missed'); three(f);
  assert.equal(f.w.view(f.id).outcome, 'confirmation_required');
});
test('low-impact action uses normal workflow without confirmation', () => {
  const w = new Workflow(); const s = w.create('information');
  assert.equal(s.outcome, 'standard_bank_workflow');
  assert.throws(() => w.issueChallenge(s.id), /Low-impact/);
});
test('real browser P-256 approval is verified and marks only eligibility', async () => {
  const f = await setup(); assert.equal(f.keys.privateKey.extractable, false);
  assert.equal((await approve(f)).outcome, 'eligible_for_bank_checks');
});
test('elevated evidence remains reviewable after a valid signature', async () => {
  const f = await setup('clone'); three(f);
  assert.equal((await approve(f)).outcome, 'confirmed_review_required');
});
test('legitimate TTS is not denied service; confirmed analyst release is available', async () => {
  const f = await setup('tts'); three(f); await approve(f);
  const s = f.w.analystDecision(f.id, { decision: 'release', reason: 'Confirmed exact action independently; accessible speech route used.' });
  assert.equal(s.outcome, 'eligible_for_bank_checks');
});
test('analyst cannot release an unconfirmed sensitive request', async () => {
  const f = await setup('clone'); three(f);
  assert.throws(() => f.w.analystDecision(f.id, { decision: 'release', reason: 'Voice seemed convincing.' }), /confirmation is required/);
});
test('callback remains a safe alternative rather than a fraud finding', async () => {
  const f = await setup('unknown');
  assert.equal(f.w.analystDecision(f.id, { decision: 'callback', reason: 'Customer offline; use registered channel recovery.' }).outcome, 'assisted_callback');
});
test('replayed valid approval is rejected', async () => {
  const f = await setup(); const c = f.w.issueChallenge(f.id).challenge;
  const response = { challenge_id: c.challenge_id, decision: 'approve', signature: await signature(f.keys, c) };
  f.w.respond(f.id, response);
  assert.throws(() => f.w.respond(f.id, response), /replay rejected/);
});
test('altered signed decision is rejected', async () => {
  const f = await setup(); const c = f.w.issueChallenge(f.id).challenge;
  const sig = await signature(f.keys, c, 'reject');
  assert.throws(() => f.w.respond(f.id, { challenge_id: c.challenge_id, decision: 'approve', signature: sig }), /Invalid signature/);
});
test('wrong customer key cannot confirm', async () => {
  const f = await setup(); const other = await setup(); const c = f.w.issueChallenge(f.id).challenge;
  assert.throws(() => f.w.respond(f.id, { challenge_id: c.challenge_id, decision: 'approve', signature: 'A'.repeat(86) }), /Invalid signature/);
  const sig = await signature(other.keys, c);
  assert.throws(() => f.w.respond(f.id, { challenge_id: c.challenge_id, decision: 'approve', signature: sig }), /Invalid signature/);
});
test('expired approval is unverified, not fraud', async () => {
  let time = 1000; const f = await setup('genuine', { now: () => time, ttl: 100 });
  const c = f.w.issueChallenge(f.id).challenge; const sig = await signature(f.keys, c);
  time = 1100;
  assert.throws(() => f.w.respond(f.id, { challenge_id: c.challenge_id, decision: 'approve', signature: sig }), /unverified does not mean fraud/);
  assert.equal(f.w.view(f.id).confirmation, 'expired');
  assert.equal(f.w.view(f.id).voice, 'unknown');
});
test('pending prompt cannot be repeatedly reissued', async () => {
  const f = await setup(); f.w.issueChallenge(f.id);
  assert.throws(() => f.w.issueChallenge(f.id), /no prompt bombing/);
});
test('changed exact action invalidates previous approval and prior release', async () => {
  const f = await setup('clone'); three(f); await approve(f);
  f.w.analystDecision(f.id, { decision: 'release', reason: 'Verified this specific action.' });
  f.w.updateAction(f.id, { type: 'increase_transfer_limit', from: 25000, to: 200000, currency: 'INR' });
  assert.equal(f.w.view(f.id).outcome, 'confirmation_and_review');
  assert.equal(f.w.view(f.id).analyst, null);
});
test('changing pending action rejects its old signed challenge', async () => {
  const f = await setup(); const c = f.w.issueChallenge(f.id).challenge; const sig = await signature(f.keys, c);
  f.w.updateAction(f.id, { type: 'increase_transfer_limit', from: 25000, to: 200000, currency: 'INR' });
  assert.throws(() => f.w.respond(f.id, { challenge_id: c.challenge_id, decision: 'approve', signature: sig }), /exact action changed/);
  const fresh = f.w.issueChallenge(f.id).challenge;
  assert.notEqual(fresh.challenge_id, c.challenge_id);
  assert.equal(fresh.action_version, 2);
});
test('response cannot be reused for another session', async () => {
  const f = await setup(); const other = await setup(); const c = f.w.issueChallenge(f.id).challenge;
  other.w.issueChallenge(other.id);
  const sig = await signature(f.keys, c);
  assert.throws(() => other.w.respond(other.id, { challenge_id: c.challenge_id, decision: 'approve', signature: sig }), /Unknown or superseded/);
});
test('customer rejection closes only this request', async () => {
  const f = await setup(); const c = f.w.issueChallenge(f.id).challenge;
  const s = f.w.respond(f.id, { challenge_id: c.challenge_id, decision: 'reject', signature: await signature(f.keys, c, 'reject') });
  assert.equal(s.outcome, 'request_cancelled'); assert.match(s.message, /no account freeze/);
});
test('cannot replace pinned credential after enrollment', async () => {
  const f = await setup(); const jwk = await webcrypto.subtle.exportKey('jwk', f.keys.publicKey);
  assert.throws(() => f.w.registerCredential(f.id, jwk), /pinned/);
});
test('private or invalid JWK is rejected', () => {
  const w = new Workflow(); const s = w.create('genuine');
  assert.throws(() => w.registerCredential(s.id, { kty: 'EC', crv: 'P-256', x: 'x', y: 'y', d: 'private' }), /public-only/);
});
test('later elevated evidence invalidates an earlier analyst release', async () => {
  const f = await setup('clone'); await approve(f);
  f.w.analystDecision(f.id, { decision: 'release', reason: 'Confirmed before acoustic evidence arrived.' });
  assert.equal(f.w.view(f.id).outcome, 'eligible_for_bank_checks');
  three(f); assert.equal(f.w.view(f.id).outcome, 'confirmed_review_required');
});
test('exact action digest changes when the requested limit changes', () => {
  assert.notEqual(actionDigest({ type: 'limit', from: 25000, to: 100000 }), actionDigest({ type: 'limit', from: 25000, to: 200000 }));
});
test('observations retain explicit simulation provenance', async () => {
  const f = await setup(); f.w.nextObservation(f.id);
  const s = f.w.view(f.id);
  assert.equal(s.provenance, 'workflow_simulation_no_audio_detector');
  assert.equal(s.observations[0].model_id, null);
  assert.equal(s.observations[0].provenance, 'simulated_fixture');
});
