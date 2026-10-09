# Live Control Center Integration Plan

## Scope

Connect the existing Indoone Backend Control Center to the existing Indoone backend. Keep all AI model/provider behavior unchanged. In particular, do not change Google AI Studio, Gemini, the local Indoone model, or model routing as part of this project.

## Required controls

- Global application intake pause/resume. The API process and this admin API stay online so an administrator can resume intake.
- Per-channel request-intake switch for WhatsApp, Instagram, Telegram, and Android.
- Global AI-reply pause/resume.
- Per-channel AI-reply switch for WhatsApp, Instagram, Telegram, and Android.
- Request counters by channel: total, successful, failed, and blocked.
- Reply counters by channel: sent, failed, and skipped while disabled.
- Recent activity records with route, event type, status, time, and HTTP code only. Do not record message bodies, tokens, phone numbers, or other message payloads.
- Server-side admin-token authorization for all read/write control endpoints.

## Implementation sequence

1. Add this plan and verify CI.
2. Add persistent control settings and privacy-safe event/metrics storage, with unit tests; wait for CI to pass.
3. Add admin-protected status/settings/metrics/activity endpoints and tests; wait for CI to pass.
4. Add a server-side intake gate that leaves health and control endpoints available and does not modify the AI provider; test it.
5. Connect per-channel reply flags, including Telegram inbound AI reply handling, while preserving provider signature/secret checks; test it.
6. Verify all backend CI/security checks; document settings required for deployment. No production server environment or running service is changed by repository commits alone.

## Safety and behavior

- A paused channel must not affect any other channel.
- Pausing application intake means application handlers are not run; the server process and control APIs remain running.
- Webhook providers must receive a safe acknowledgement while intake is paused, so they do not retry a backlog that could later trigger unexpected replies. Paused webhook events are dropped, not queued.
- Reply-off does not mean intake-off: requests can still be counted, while automated AI replies are skipped.
- Existing authentication and provider webhook signature/secret validation must remain in place.
- Do not introduce a new AI provider, AI Studio key, model selection, model routing, or changes to model generation.
- Before production deployment, configure a separate high-entropy `INDOONE_CONTROL_CENTER_ADMIN_TOKEN` on the backend host, configure the public frontend API origin, allow only the intended Control Center origin in backend CORS, and keep the token out of source code and frontend build variables.

## Acceptance checklist

- [ ] Unit/API tests cover default settings, persistence, admin auth, and invalid setting payloads.
- [ ] Global intake off blocks all application API handling while admin controls and health checks stay available.
- [ ] Each channel's intake and reply settings are independent.
- [ ] Metrics distinguish request totals/success/failure/blocked and replies sent/failed/skipped.
- [ ] Dashboard reads real server state and metrics; it does not display sample metrics as live.
- [ ] Existing model/provider logic is unchanged.
- [ ] Backend CI and security smoke check are green.
