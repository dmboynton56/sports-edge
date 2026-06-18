#!/usr/bin/env python3
"""Cloud Run HTTP bridge from Cloud Scheduler to GitHub workflow_dispatch."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


OWNER = os.getenv("GITHUB_OWNER", "dmboynton56")
REPO = os.getenv("GITHUB_REPO", "sports-edge")
REF = os.getenv("GITHUB_REF", "main")
PORT = int(os.getenv("PORT", "8080"))

WORKFLOWS: dict[str, dict[str, Any]] = {
    "daily-refresh": {
        "workflow": "daily-refresh.yml",
        "inputs": {
            "dry_run": "false",
            "force_full_rebuild": "false",
        },
    },
    "world-cup-refresh": {
        "workflow": "world-cup-refresh.yml",
        "inputs": {
            "dry_run": "false",
            "n_sims": "50000",
        },
    },
    "player-markets-refresh": {
        "workflow": "player-markets-refresh.yml",
        "inputs": {
            "date": "",
            "run_pga": "true",
            "run_mlb_hr": "false",
            "train_mlb_hr": "false",
            "sync_supabase": "true",
            "sync_bigquery": "true",
        },
    },
}


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("content-length") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _dispatch(workflow: str, inputs: dict[str, str]) -> tuple[int, str]:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not configured")

    url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/workflows/{workflow}/dispatches"
    payload = json.dumps({"ref": REF, "inputs": inputs}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "sports-edge-cloud-scheduler",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.status), response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        raise RuntimeError(f"GitHub dispatch failed with {exc.code}: {body}") from exc


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/healthz":
            _json_response(self, 200, {"ok": True})
            return
        _json_response(self, 404, {"error": "not_found"})

    def do_POST(self) -> None:
        slug = self.path.strip("/").removeprefix("dispatch/")
        config = WORKFLOWS.get(slug)
        if not config:
            _json_response(self, 404, {"error": "unknown_workflow", "workflow": slug})
            return

        try:
            body = _read_body(self)
            inputs = dict(config["inputs"])
            override_inputs = body.get("inputs") if isinstance(body, dict) else None
            if isinstance(override_inputs, dict):
                inputs.update({str(key): str(value) for key, value in override_inputs.items()})
            status, response_body = _dispatch(str(config["workflow"]), inputs)
        except Exception as exc:  # noqa: BLE001
            _json_response(self, 500, {"error": str(exc), "workflow": slug})
            return

        _json_response(
            self,
            202,
            {
                "github_status": status,
                "github_body": response_body,
                "workflow": slug,
                "inputs": inputs,
            },
        )

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Listening on :{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
