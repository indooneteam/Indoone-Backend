# Release hardening — 2026-09-16

This batch groups the remaining high-impact backend hardening work into one release cycle:

- connector/tool registry execution coverage checks
- DOCX archive expansion and archive-entry safety limits
- B2 object-key validation against traversal/control-character abuse
- request IDs preserved on early and handled API errors
- existing SQLite WAL/busy-timeout and single-use OAuth-state protections retained
- CI remains the release gate for the final `develop` SHA
