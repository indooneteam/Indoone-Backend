"""The deploy workflow must stay the deploy we run by hand.

A workflow is code that only runs in production, which makes it the easiest thing in the
repository to break and the last thing anyone notices. These tests hold the four properties
that matter, and nothing about the YAML's wording:

1. it parses, and it runs on a push to main;
2. it runs `scripts/ops/deploy.sh` - the same script a person runs on the box;
3. it refuses to start when it is not configured, instead of failing halfway;
4. it carries no credential of its own.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(".github/workflows/deploy.yml")


@pytest.fixture(scope="module")
def workflow() -> dict:
    assert WORKFLOW.is_file(), "the deploy workflow is missing"
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_it_parses_and_is_named(workflow: dict) -> None:
    assert workflow["name"] == "Deploy Backend"
    assert "deploy" in workflow["jobs"]


def test_it_runs_on_a_push_to_main(workflow: dict) -> None:
    triggers = workflow.get("on") or workflow.get(True)
    push = triggers["push"]

    assert push["branches"] == ["main"]
    assert "app/**" in push["paths"], "a backend change must deploy"
    assert "workflow_dispatch" in triggers, "a person must be able to run it by hand"


def test_it_runs_the_same_script_a_person_runs(workflow: dict) -> None:
    """The point of the automation is that it is not a second way to deploy."""
    steps = workflow["jobs"]["deploy"]["steps"]
    commands = " ".join(str(step.get("run", "")) for step in steps)

    assert "scripts/ops/deploy.sh" in commands
    assert "/home/ubuntu/Indoone-Backend" in commands


def test_it_stops_before_deploying_when_it_is_not_configured(workflow: dict) -> None:
    steps = workflow["jobs"]["deploy"]["steps"]
    preflight = [step for step in steps if step.get("id") == "preflight"]

    assert preflight, "a deploy with no configuration must say so, not fail halfway"
    script = preflight[0]["run"]
    assert "LIGHTSAIL_HOST" in script
    assert "LIGHTSAIL_SSH_KEY" in script
    assert "exit 1" in script


def test_it_confirms_the_commit_that_is_actually_running(workflow: dict) -> None:
    steps = workflow["jobs"]["deploy"]["steps"]
    names = " ".join(str(step.get("name", "")) for step in steps)

    assert "commit" in names.lower()


def test_two_deploys_cannot_overlap(workflow: dict) -> None:
    assert workflow["concurrency"]["group"] == "deploy-production"
    assert workflow["concurrency"]["cancel-in-progress"] is False


def test_it_carries_no_credential_of_its_own() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "BEGIN OPENSSH PRIVATE KEY" not in text
    assert "BEGIN RSA PRIVATE KEY" not in text
    assert "ssh-rsa AAAA" not in text
    assert "${{ secrets." in text, "credentials come from secrets, never from the file"


def test_it_asks_for_the_least_it_needs(workflow: dict) -> None:
    assert workflow["permissions"] == {"contents": "read"}
