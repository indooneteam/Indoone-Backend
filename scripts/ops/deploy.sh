#!/usr/bin/env bash
#
# Deploy the backend to this box, and prove it came up.
#
# This exists because the one-line deploy had two holes that each cost us an outage:
#
#  1. It never copied the systemd unit. When the entry point moved to
#     scripts/ops/run_server.py, the unit on the box still ran the old module name and
#     the service restart-looped until someone noticed.
#  2. It waited a fixed three seconds and then asked systemctl whether the service was
#     active. A unit in a restart loop reports "activating", not "active", so the check
#     failed and the && chain stopped - which meant the health check never ran, and the
#     output looked like a quiet success.
#
# So: pull, install, sync the unit if it changed, restart, wait for active up to the
# deadline, then curl the health endpoint. If any of that fails, print the journal and
# exit non-zero.
#
# Run it as:  bash /home/ubuntu/Indoone-Backend/scripts/ops/deploy.sh
#
set -euo pipefail

REPO="${INDOONE_DEPLOY_REPO:-/home/ubuntu/Indoone-Backend}"
UNIT="${INDOONE_DEPLOY_UNIT:-/etc/systemd/system/indoone-backend.service}"
SERVICE="${INDOONE_DEPLOY_SERVICE:-indoone-backend}"
HEALTH="${INDOONE_DEPLOY_HEALTH:-http://localhost:8000/health}"
BRANCH="${INDOONE_DEPLOY_BRANCH:-main}"
WAIT_SECONDS="${INDOONE_DEPLOY_WAIT_SECONDS:-45}"

cd "$REPO"

echo "==> pulling $BRANCH"
git pull origin "$BRANCH"

echo "==> installing requirements"
"$REPO/.venv/bin/pip" install -q -r requirements.txt

# The unit in the repository is the source of truth. Copy it only when it differs, so a
# code deploy does not touch systemd needlessly, and a unit change never goes missing.
if [ -f deploy/indoone-backend.service ]; then
    if ! sudo cmp -s deploy/indoone-backend.service "$UNIT"; then
        echo "==> unit changed, installing $UNIT"
        sudo cp deploy/indoone-backend.service "$UNIT"
        sudo systemctl daemon-reload
    fi
fi

echo "==> restarting $SERVICE"
sudo systemctl restart "$SERVICE"

# Wait for the service to settle. "activating" is not success and is not failure yet.
state=""
deadline=$((SECONDS + WAIT_SECONDS))
while [ "$SECONDS" -lt "$deadline" ]; do
    state="$(systemctl is-active "$SERVICE" || true)"
    if [ "$state" = "active" ]; then
        break
    fi
    sleep 1
done

if [ "$state" != "active" ]; then
    echo "!! $SERVICE is '$state' after ${WAIT_SECONDS}s, not active" >&2
    echo "--- last 40 journal lines ---" >&2
    sudo journalctl -u "$SERVICE" -n 40 --no-pager >&2 || true
    exit 1
fi

echo "==> $SERVICE is active"

# The real check: does it answer?
if ! response="$(curl -fsS "$HEALTH")"; then
    echo "!! health check failed: $HEALTH" >&2
    sudo journalctl -u "$SERVICE" -n 40 --no-pager >&2 || true
    exit 1
fi

echo "==> health: $response"
echo "==> deploy finished"
