#!/usr/bin/env python3
"""
Docker entrypoint — loads secrets from AWS Secrets Manager and SSM
Parameter Store, writes credential files, then exec's the target service.

Runs as PID 1 in each container. The SERVICE_TYPE env var determines
which service to start: backend, services, or mcp.
"""

import os
import sys
import json
import base64
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("entrypoint")


def load_secrets():
    """Load secrets from AWS Secrets Manager and SSM into env + files."""
    import boto3

    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    sm = boto3.client("secretsmanager", region_name=region)
    ssm = boto3.client("ssm", region_name=region)

    env_vars = {}

    # ── SSM Parameters (/roseboro/env/*) → env vars ──────────────
    try:
        paginator = ssm.get_paginator("get_parameters_by_path")
        for page in paginator.paginate(
            Path="/roseboro/env/", Recursive=True, WithDecryption=True
        ):
            for param in page["Parameters"]:
                key = param["Name"].split("/")[-1]
                env_vars[key] = param["Value"]
        log.info(f"Loaded {len(env_vars)} SSM parameters")
    except Exception as e:
        log.warning(f"Failed to load SSM parameters: {e}")

    # ── Secrets Manager → env vars + files ────────────────────────
    secret_map = {
        "roseboro/outlook-client-secret": "_single_OUTLOOK_CLIENT_SECRET",
        "roseboro/hubspot-api-key": "_single_HUBSPOT_API_KEY",
        "roseboro/anthropic-api-key": "_single_ANTHROPIC_API_KEY",
        "roseboro/slack-tokens": "_dict",
        "roseboro/ringcentral": "_dict",
        "roseboro/icloud": "_dict_icloud",
        "roseboro/misc-api-keys": "_dict",
        "roseboro/pushover": "_dict",
        "roseboro/google-tokens": "_google",

        "roseboro/apns-auth-key": "_apns",
        "roseboro/couchdb-credentials": "_dict_couchdb",
    }

    for secret_name, handler in secret_map.items():
        try:
            resp = sm.get_secret_value(SecretId=secret_name)
            raw = resp["SecretString"]
            try:
                value = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                value = raw

            if handler.startswith("_single_"):
                key = handler[len("_single_"):]
                env_vars[key] = value if isinstance(value, str) else value.get(key, "")

            elif handler == "_dict":
                if isinstance(value, dict):
                    env_vars.update({k: str(v) for k, v in value.items()})

            elif handler == "_dict_icloud":
                if isinstance(value, dict):
                    env_vars["ICLOUD_USERNAME"] = value.get("username", "")
                    env_vars["ICLOUD_PASSWORD"] = value.get("password", "")

            elif handler == "_dict_couchdb":
                if isinstance(value, dict):
                    env_vars["COUCHDB_USER"] = value.get("username", value.get("COUCHDB_USER", ""))
                    env_vars["COUCHDB_PASSWORD"] = value.get("password", value.get("COUCHDB_PASSWORD", ""))

            elif handler == "_google":
                _write_google_creds(value)

            elif handler == "_apns":
                _write_apns_key(value, env_vars)

            log.info(f"Loaded secret: {secret_name}")
        except Exception as e:
            log.warning(f"Failed to load secret {secret_name}: {e}")

    # ── Teller mTLS certificate ────────────────────────────────
    _write_teller_certs(sm)

    # ── Write .env file ───────────────────────────────────────────
    env_path = "/app/.env"
    fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        for k, v in sorted(env_vars.items()):
            # Escape newlines and quotes for .env format
            safe_v = str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            f.write(f'{k}="{safe_v}"\n')

    # Set in current process environment (inherited by exec'd service)
    for k, v in env_vars.items():
        os.environ[k] = str(v)

    log.info(f"Loaded {len(env_vars)} total environment variables")


def _write_google_creds(value):
    """Write Google OAuth pickle files from base64-encoded secret."""
    if not isinstance(value, dict):
        return
    creds_dir = "/app/google_creds"
    os.makedirs(creds_dir, exist_ok=True)
    count = 0
    for token_name, token_b64 in value.items():
        if not token_name.endswith(".pickle"):
            continue
        try:
            token_path = os.path.join(creds_dir, token_name)
            with open(token_path, "wb") as f:
                f.write(base64.b64decode(token_b64))
            count += 1
        except Exception as e:
            log.warning(f"Failed to write {token_name}: {e}")
    log.info(f"Wrote {count} Google credential files")



def _write_apns_key(value, env_vars):
    """Write APNS authentication key to file."""
    key_dir = "/app/certs"
    os.makedirs(key_dir, exist_ok=True)
    key_path = os.path.join(key_dir, "AuthKey.p8")
    key_data = value if isinstance(value, str) else value.get("key", "")
    with open(key_path, "w") as f:
        f.write(key_data)
    os.chmod(key_path, 0o600)
    env_vars["APNS_AUTH_KEY_PATH"] = key_path


def _write_teller_certs(sm):
    """Write Teller mTLS certificate and private key to /app/certs/teller/."""
    teller_dir = "/app/certs/teller"
    os.makedirs(teller_dir, exist_ok=True)
    for secret_name, filename in [
        ("roseboro/teller/certificate", "certificate.pem"),
        ("roseboro/teller/private-key", "private_key.pem"),
    ]:
        try:
            resp = sm.get_secret_value(SecretId=secret_name)
            path = os.path.join(teller_dir, filename)
            with open(path, "w") as f:
                f.write(resp["SecretString"])
            os.chmod(path, 0o600)
            log.info(f"Wrote Teller cert: {path}")
        except Exception as e:
            log.warning(f"Failed to load Teller cert {secret_name}: {e}")


def main():
    log.info("Docker entrypoint starting...")

    # Load secrets (skip in local dev if AWS creds aren't available)
    try:
        load_secrets()
    except Exception as e:
        log.error(f"Secret loading failed: {e}")
        log.info("Continuing with existing environment variables...")

    service_type = os.environ.get("SERVICE_TYPE", "backend")
    log.info(f"Starting service: {service_type}")

    if service_type == "backend":
        os.execvp("python3", [
            "python3", "-m", "uvicorn", "app:app",
            "--host", "0.0.0.0",
            "--port", "8081",
            "--log-level", "info",
            "--timeout-keep-alive", "30",
            "--workers", "1",
        ])

    elif service_type == "services":
        os.execvp("supervisord", [
            "supervisord", "-n", "-c", "/etc/supervisor/conf.d/services.conf",
        ])

    elif service_type == "mcp":
        os.execvp("python3", [
            "python3", "mcp_server.py",
            "--transport", "streamable-http",
            "--port", "8100",
        ])

    else:
        log.error(f"Unknown SERVICE_TYPE: {service_type}")
        sys.exit(1)


if __name__ == "__main__":
    main()
