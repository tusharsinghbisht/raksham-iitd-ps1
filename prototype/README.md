# Workflow prototype

A local Node.js simulation of EchoLock's **Confirm** and **Review** layers: the policy engine, the signed exact-action confirmation and the analyst console. No dependencies; Node 20+.

```
npm test     # 26 tests
npm start    # http://localhost:8787
```

In the demo you play all three roles on one page:

1. **Bank agent:** pick a scenario (consented clone, genuine caller, legitimate TTS user, poor audio, missed clone, low-impact request), start a session and emit voice windows. Three high windows latch the call as ELEVATED; one does not.
2. **Customer app:** the browser holds a non-extractable P-256 key, pinned at "enrolment". The challenge shows the exact change. Approve or reject signs it, and the server verifies the signature, action digest, nonce, expiry and single use.
3. **Analyst:** an ELEVATED case stays open for review even after a valid approval. Release needs a written reason and a confirmed request.

Every step lands in a reason-code timeline held in RAM only.

What the tests cover: confirmation is required even with no audio, UNKNOWN never counts as genuine, and a missed clone still can't proceed. Replayed, altered, expired, cross-session or wrong-key approvals are refused. A changed action voids earlier approvals and releases. A legitimate TTS user isn't denied service.

**Not real yet:** the voice windows are fixture scores. The detector from [`../eval`](../eval) isn't wired in, and there is no telephony, bank core or real biometric attestation.
