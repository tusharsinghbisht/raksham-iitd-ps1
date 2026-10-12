import { createHash, createPublicKey, randomBytes, randomUUID, verify } from 'node:crypto';

export const DOMAIN = 'echolock.action-confirm.v1';
export const SCENARIOS = Object.freeze({
  clone: { name: 'Consented-clone scenario (simulated)', scores: [0.91, 0.96, 0.94], sensitive: true },
  genuine: { name: 'Genuine caller (simulated)', scores: [0.12, 0.18, 0.15], sensitive: true },
  tts: { name: 'Legitimate TTS / assistive speech (simulated)', scores: [0.93, 0.97, 0.95], sensitive: true },
  unknown: { name: 'Poor audio / detector unavailable (simulated)', scores: [null, null, null], sensitive: true },
  missed: { name: 'Clone missed by detector (simulated)', scores: [0.15, 0.19, 0.12], sensitive: true },
  information: { name: 'Low-impact information request (simulated)', scores: [0.1, 0.2, 0.1], sensitive: false },
});

export class WorkflowError extends Error {
  constructor(message, status = 400) { super(message); this.status = status; }
}
const hash = value => createHash('sha256').update(value).digest('hex');
export const actionDigest = action => hash(JSON.stringify([
  action.type, action.from ?? null, action.to ?? null, action.currency ?? null,
]));

// Both ends use this exact ordered array, encoded as UTF-8 JSON.
export const confirmationMessage = (challenge, decision) => JSON.stringify([
  DOMAIN, challenge.challenge_id, challenge.customer_id, challenge.session_id,
  challenge.credential_id, challenge.action_version, challenge.action_digest,
  challenge.nonce, challenge.expires_at, decision,
]);

export class Workflow {
  constructor({ now = Date.now, ttl = 60_000 } = {}) {
    this.now = now;
    this.ttl = ttl;
    this.sessions = new Map();
  }

  log(session, type, detail) {
    session.events.push({ at: this.now(), type, detail });
  }

  session(id) {
    const s = this.sessions.get(id);
    if (!s) throw new WorkflowError('Unknown demo session', 404);
    return s;
  }

  create(scenario) {
    if (!Object.hasOwn(SCENARIOS, scenario)) throw new WorkflowError('Unknown scenario');
    if (this.sessions.size >= 100) throw new WorkflowError('Demo capacity reached; restart the local server', 429);
    const fixture = SCENARIOS[scenario];
    const action = fixture.sensitive
      ? { type: 'increase_transfer_limit', from: 25000, to: 100000, currency: 'INR' }
      : { type: 'general_information' };
    const s = {
      id: randomUUID(), customer_id: 'consenting-demo-customer', scenario,
      action, action_version: 1, credential: null, challenge: null,
      observations: [], fixture_index: 0, elevated: false, confirmation: 'not_requested',
      confirmed_digest: null, confirmed_version: null,
      analyst: null, cancelled: false, events: [],
    };
    this.sessions.set(s.id, s);
    this.log(s, 'SESSION_CREATED', 'Local demo only; acoustic scores are simulated. No bank action is executed.');
    return this.view(s.id);
  }

  registerCredential(id, jwk) {
    const s = this.session(id);
    if (s.challenge || s.credential) throw new WorkflowError('Demo credential is pinned for this session', 409);
    if (!jwk || jwk.kty !== 'EC' || jwk.crv !== 'P-256' || typeof jwk.x !== 'string'
      || typeof jwk.y !== 'string' || Object.hasOwn(jwk, 'd')) {
      throw new WorkflowError('Expected a public-only P-256 JWK');
    }
    let key;
    try { key = createPublicKey({ key: { kty: jwk.kty, crv: jwk.crv, x: jwk.x, y: jwk.y }, format: 'jwk' }); }
    catch { throw new WorkflowError('Invalid public key'); }
    const credential_id = hash(key.export({ type: 'spki', format: 'der' }));
    s.credential = { credential_id, key };
    this.log(s, 'DEMO_KEY_REGISTERED', 'Browser public key pinned. This is not bank enrollment or biometric verification.');
    return this.view(id);
  }

  nextObservation(id) {
    const s = this.session(id);
    if (s.cancelled) throw new WorkflowError('Request is closed', 409);
    const fixture = SCENARIOS[s.scenario];
    if (s.fixture_index >= fixture.scores.length) throw new WorkflowError('All three fixture windows have been emitted', 409);
    const score = fixture.scores[s.fixture_index++];
    s.observations.push({
      sequence: s.fixture_index,
      end_ms: 4000 + (s.fixture_index - 1) * 2000,
      quality: score === null ? 'insufficient' : 'usable',
      score, provenance: 'simulated_fixture', model_id: null,
    });
    const usable = s.observations.filter(o => o.quality === 'usable').slice(-3);
    if (usable.length === 3) {
      const ordered = usable.map(o => o.score).sort((a, b) => a - b);
      if (ordered[1] >= 0.75 && usable.filter(o => o.score >= 0.80).length >= 2) {
        if (!s.elevated) {
          // Newly arriving high evidence invalidates any earlier analyst release.
          s.analyst = null;
          this.log(s, 'ELEVATED_VOICE_SUSPICION', 'Repeated simulated evidence; not proof of fraud. Analyst review is required for a sensitive action.');
        }
        s.elevated = true;
      }
    }
    this.log(s, 'SIMULATED_OBSERVATION', score === null ? 'Unknown: no usable acoustic observation' : `Fixture signal ${score.toFixed(2)}; not fraud probability`);
    return this.view(id);
  }

  issueChallenge(id) {
    const s = this.session(id);
    if (s.cancelled) throw new WorkflowError('Request is closed', 409);
    if (!SCENARIOS[s.scenario].sensitive) throw new WorkflowError('Low-impact action uses standard workflow');
    if (!s.credential) throw new WorkflowError('Register the demo customer key first');
    if (s.confirmation === 'approved') throw new WorkflowError('This exact action is already confirmed', 409);
    if (s.challenge && !s.challenge.consumed && this.now() < s.challenge.expires_at
      && s.challenge.action_version === s.action_version && s.challenge.action_digest === actionDigest(s.action)) {
      throw new WorkflowError('A confirmation is already pending; no prompt bombing', 409);
    }
    if (s.events.filter(e => e.type === 'CHALLENGE_ISSUED').length >= 5) {
      throw new WorkflowError('Demo prompt limit reached; use assisted recovery', 429);
    }
    s.challenge = {
      domain: DOMAIN, challenge_id: randomUUID(), customer_id: s.customer_id,
      session_id: s.id, credential_id: s.credential.credential_id,
      action_version: s.action_version, action_digest: actionDigest(s.action),
      nonce: randomBytes(32).toString('base64url'), expires_at: this.now() + this.ttl,
      consumed: false,
    };
    s.confirmation = 'pending';
    this.log(s, 'CHALLENGE_ISSUED', 'Confirm the exact action through the independent demo customer panel.');
    return this.view(id);
  }

  respond(id, { challenge_id, decision, signature } = {}) {
    const s = this.session(id);
    const c = s.challenge;
    if (!c || c.challenge_id !== challenge_id) throw new WorkflowError('Unknown or superseded challenge', 409);
    if (c.consumed) throw new WorkflowError('Challenge was already consumed; replay rejected', 409);
    if (this.now() >= c.expires_at) {
      s.confirmation = 'expired';
      throw new WorkflowError('Challenge expired; unverified does not mean fraud', 409);
    }
    if (s.cancelled) throw new WorkflowError('Request is closed', 409);
    if (c.action_version !== s.action_version || c.action_digest !== actionDigest(s.action)
      || c.customer_id !== s.customer_id || c.session_id !== s.id
      || c.credential_id !== s.credential?.credential_id) {
      throw new WorkflowError('Customer, credential, session or exact action changed', 409);
    }
    if (!['approve', 'reject'].includes(decision)) throw new WorkflowError('Invalid confirmation decision');
    if (typeof signature !== 'string' || !/^[A-Za-z0-9_-]{86}$/.test(signature)) throw new WorkflowError('Expected a 64-byte P-256 signature');
    const bytes = Buffer.from(signature, 'base64url');
    let valid = false;
    try {
      valid = verify('sha256', Buffer.from(confirmationMessage(c, decision)), {
        key: s.credential.key, dsaEncoding: 'ieee-p1363',
      }, bytes);
    } catch { valid = false; }
    if (!valid) {
      this.log(s, 'INVALID_CONFIRMATION', 'Signature rejected. This is not a caller-fraud classification.');
      throw new WorkflowError('Invalid signature or altered decision', 403);
    }
    c.consumed = true;
    s.confirmation = decision === 'approve' ? 'approved' : 'rejected';
    s.confirmed_digest = decision === 'approve' ? c.action_digest : null;
    s.confirmed_version = decision === 'approve' ? c.action_version : null;
    if (decision === 'reject') s.cancelled = true;
    this.log(s, 'CONFIRMATION_' + s.confirmation.toUpperCase(), decision === 'approve'
      ? 'Exact action confirmed by this demo key; not proof of caller identity. Bank checks still required.'
      : 'Customer declined this request; no account freeze.');
    return this.view(id);
  }

  // Mock of a trusted BANK action-context update, not a caller/client score input.
  updateAction(id, action) {
    const s = this.session(id);
    if (s.cancelled) throw new WorkflowError('Request is closed', 409);
    s.action = { ...action };
    s.action_version += 1;
    s.confirmation = 'not_requested';
    s.confirmed_digest = null;
    s.confirmed_version = null;
    s.analyst = null;
    this.log(s, 'BANK_ACTION_CHANGED', 'Exact action changed; prior confirmation cannot authorise it.');
    return this.view(id);
  }

  analystDecision(id, { decision, reason } = {}) {
    const s = this.session(id);
    if (s.cancelled) throw new WorkflowError('Request is closed', 409);
    if (!['release', 'callback', 'cancel'].includes(decision)) throw new WorkflowError('Unknown analyst decision');
    if (typeof reason !== 'string' || reason.trim().length < 8 || reason.length > 500) {
      throw new WorkflowError('Supply an 8–500 character review reason without sensitive data');
    }
    if (decision === 'release' && SCENARIOS[s.scenario].sensitive && !this.isConfirmed(s)) {
      throw new WorkflowError('Valid exact-action confirmation is required; use assisted callback otherwise', 409);
    }
    s.analyst = { decision, reason: reason.trim(), at: this.now() };
    if (decision === 'cancel') s.cancelled = true;
    this.log(s, 'ANALYST_' + decision.toUpperCase(), reason.trim());
    return this.view(id);
  }

  isConfirmed(s) {
    return s.confirmation === 'approved' && s.confirmed_version === s.action_version
      && s.confirmed_digest === actionDigest(s.action);
  }

  view(id) {
    const s = this.session(id);
    const sensitive = SCENARIOS[s.scenario].sensitive;
    const usable = s.observations.filter(o => o.quality === 'usable');
    const voice = s.elevated ? 'elevated' : usable.length >= 3 ? 'not_elevated' : 'unknown';
    let confirmation = s.confirmation;
    if (confirmation === 'pending' && this.now() >= s.challenge.expires_at) confirmation = 'expired';
    let outcome, message;
    if (s.cancelled) {
      outcome = 'request_cancelled'; message = 'This request was cancelled; no account freeze or real bank action.';
    } else if (!sensitive) {
      outcome = 'standard_bank_workflow'; message = 'Low-impact request: continue normal bank workflow.';
    } else if (s.analyst?.decision === 'callback') {
      outcome = 'assisted_callback'; message = 'Use the previously registered channel or staffed recovery; timeout is not fraud proof.';
    } else if (this.isConfirmed(s)) {
      if (s.elevated && s.analyst?.decision !== 'release') {
        outcome = 'confirmed_review_required'; message = 'Exact action confirmed; voice suspicion still requires human review.';
      } else {
        outcome = 'eligible_for_bank_checks'; message = 'Eligible for existing bank authentication and authorisation checks. Nothing was executed.';
      }
    } else if (s.elevated) {
      outcome = 'confirmation_and_review'; message = 'Possible voice impersonation, not proof. Confirm exact action and obtain review.';
    } else {
      outcome = 'confirmation_required'; message = 'Sensitive action requires independent exact-action confirmation, even if detection misses a clone.';
    }
    return {
      id: s.id, customer_id: s.customer_id, scenario: s.scenario, name: SCENARIOS[s.scenario].name,
      action: { ...s.action }, action_version: s.action_version, sensitive,
      registered: Boolean(s.credential), voice, confirmation, outcome, message,
      challenge: s.challenge ? { ...s.challenge } : null,
      observations: s.observations.map(o => ({ ...o })),
      analyst: s.analyst ? { ...s.analyst } : null, events: s.events.map(e => ({ ...e })),
      provenance: 'workflow_simulation_no_audio_detector',
    };
  }
}
