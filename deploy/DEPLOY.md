# Deploying the Indoone backend

The backend runs on an AWS Lightsail instance as a systemd service, from a virtualenv in
the checkout. This note exists because until now the service definition and the environment
lived only on that machine: if the instance were lost, the setup would have been lost with
it. `deploy/indoone-backend.service` is now the copy in the repository.

## What lives where

| | |
|---|---|
| Project checkout | `/home/ubuntu/Indoone-Backend` |
| Virtualenv | `/home/ubuntu/Indoone-Backend/.venv` |
| Environment file | `/etc/indoone/indoone.env` |
| Service unit | `/etc/systemd/system/indoone-backend.service` |
| Entry point | `python -m scripts.ops.run_server` |
| Health check | `http://localhost:8000/health` |

**The environment file is not in this repository, and must not be.** It holds the API keys,
the database paths and the Fernet key. `.env.example` lists every variable the code reads,
with placeholder values; that file *is* in the repository and is the reference for what
`/etc/indoone/indoone.env` needs.

## First-time setup on a fresh instance

```bash
sudo apt-get update
sudo apt-get install -y python3-venv git

git clone https://github.com/indooneteam/Indoone-Backend.git /home/ubuntu/Indoone-Backend
cd /home/ubuntu/Indoone-Backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

sudo mkdir -p /etc/indoone
sudo cp .env.example /etc/indoone/indoone.env
sudo chmod 600 /etc/indoone/indoone.env
sudo nano /etc/indoone/indoone.env      # fill in the real values

sudo cp deploy/indoone-backend.service /etc/systemd/system/indoone-backend.service
sudo systemctl daemon-reload
sudo systemctl enable --now indoone-backend
```

## Deploying a new version

One line, and it can be pasted as it is:

```bash
cd /home/ubuntu/Indoone-Backend && git pull origin main && /home/ubuntu/Indoone-Backend/.venv/bin/pip install -q -r requirements.txt && sudo systemctl restart indoone-backend && sleep 3 && systemctl is-active indoone-backend && curl -s http://localhost:8000/health
```

The `pip install` step matters when `requirements.txt` changes, which it does: it gained
`defusedxml` in the change that hardened the RSS parsing.

## Reading the environment without printing the secrets

To see which variables are set, without their values:

```bash
sudo sed 's/=.*/=<set>/' /etc/indoone/indoone.env | sort
```

To see one variable's value when it is not a secret:

```bash
sudo grep '^INDOONE_MODEL_BACKEND=' /etc/indoone/indoone.env
```

Do not `cat` the file, and do not paste its output anywhere. It contains the keys.

## If the service will not start

```bash
systemctl status indoone-backend --no-pager
sudo journalctl -u indoone-backend -n 50 --no-pager
```

The first thing to check is that the service is reading the environment file at all: a unit
without `EnvironmentFile=` starts and then fails on the first request that needs a key.
