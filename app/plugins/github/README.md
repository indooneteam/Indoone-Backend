# GitHub plugin

The GitHub integration is organized here for provider-specific repository, issue, and pull-request operations. OAuth state and encrypted per-user token storage use the shared integration registry so the existing API URLs remain backward-compatible.

## Backend configuration

Set these values in the backend secret store; never commit them:

- `INDOONE_GITHUB_CLIENT_ID`
- `INDOONE_GITHUB_CLIENT_SECRET`
- `INDOONE_OAUTH_ENCRYPTION_KEY` (shared Fernet key used for encrypted integration tokens)
- `INDOONE_CAPABILITY_DB` (persistent SQLite path in production)

The OAuth app requests the valid GitHub OAuth scopes `read:user public_repo`. This intentionally supports public repositories and avoids silently granting blanket write access to private repositories. Accessing private repositories with granular read-only permissions should use a GitHub App installation flow; a classic OAuth app's `repo` scope is broader and should not be enabled without explicit approval.

OAuth starts at `POST /api/integrations/github/connect`, returns through `GET /api/integrations/github/callback`, and exposes connection metadata at `GET /api/integrations/github/status?user_id=...`. Repository, issue, and pull-request reads use the stable paths under `/api/integrations/github/`. Issue creation and comments continue to require `approved=true`.

A green CI run validates code and mocked provider behavior. A live connection still requires the OAuth app credentials, an encrypted-token database, and user authorization on the deployed environment.
