#!/usr/bin/env python3
import json
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise SystemExit(f"CONTRACT ERROR: {message}")


def transition(gate, target):
    return next(item for item in gate["transitions"] if item["to"] == target)


def main():
    experiment = load(ROOT / "experiment-envelope.schema.json")
    provider = load(ROOT / "model-provider.schema.json")
    tool = load(ROOT / "scientific-tool.schema.json")
    receipt = load(ROOT / "receipt.schema.json")
    gate = load(ROOT / "claim-gate.json")
    runtime = load(ROOT / "runtime-policy.json")
    architecture = load(ROOT.parent / "architecture.json")
    security = load(REPO / "SECURITY.md")

    provider_ref = experiment["properties"]["provider"]["$ref"]
    require(urljoin(experiment["$id"], provider_ref) == provider["$id"], "provider $ref does not resolve to model-provider $id")
    require(experiment["properties"]["profile"].get("const") == "NS-LLM", "experiment profile is not canonical NS-LLM")
    require(runtime.get("default_profile") == "NS-LLM", "runtime profile is not canonical NS-LLM")
    require(architecture.get("profile") == "NS-LLM", "architecture profile is not canonical NS-LLM")

    seed_rules = experiment.get("allOf", [])
    require(any(rule.get("then", {}).get("required") == ["seed"] for rule in seed_rules), "provider seed capability does not require envelope seed")

    required_tool = set(tool.get("required", []))
    require({"determinism", "receipt_required"} <= required_tool, "tool determinism/receipt requirements are not mandatory")
    require(tool["properties"]["receipt_required"].get("const") is True, "receipt_required is not const true")
    expected_bindings = {
        "solver": "NUMERICAL_EXECUTION",
        "symbolic": "SYMBOLIC_DERIVATION",
        "formal": "FORMAL_CHECK",
        "retrieval": "RETRIEVED_SOURCE",
    }
    bindings = {}
    for rule in tool.get("allOf", []):
        class_schema = rule.get("if", {}).get("properties", {}).get("class", {})
        evidence = rule.get("then", {}).get("properties", {}).get("evidence_class", {}).get("const")
        if "const" in class_schema and evidence:
            bindings[class_schema["const"]] = evidence
    require(all(bindings.get(k) == v for k, v in expected_bindings.items()), "privileged tool/evidence bindings are incomplete")

    actor = receipt["properties"]["actor"]
    require(actor.get("additionalProperties") is False, "receipt actor must reject undeclared fields")
    require(actor["properties"]["id"].get("minLength") == 1, "receipt actor id must be nonempty")
    numerical_rule = next((rule for rule in receipt.get("allOf", []) if rule.get("if", {}).get("properties", {}).get("evidence_class", {}).get("const") == "NUMERICAL_EXECUTION"), None)
    require(numerical_rule is not None and "environment" in numerical_rule.get("then", {}).get("required", []), "numerical receipts must require environment")
    environment = receipt["properties"]["environment"]
    require({"runtime_id", "runtime_version", "platform"} <= set(environment.get("required", [])), "execution environment identity is incomplete")

    valid_receipt = gate.get("predicates", {}).get("VALID_RECEIPT", {})
    require(valid_receipt.get("requires", {}).get("status") == "ok", "VALID_RECEIPT must require status ok")
    for target in ("INFERENCE_ONLY", "SOURCE_SUPPORTED", "EXECUTION_SUPPORTED", "FORMALLY_CHECKED"):
        require("VALID_RECEIPT" in transition(gate, target).get("requires", []), f"{target} upgrade lacks receipt requirement")
    require("non_ok_receipt_status" in gate.get("fail_closed_on", []), "claim gate does not fail closed on non-ok receipts")

    private = security.get("reporting", {}).get("private_channel", {})
    require(private.get("kind") == "email" and private.get("address") == "trent@qsol-imc.com", "private security reporting channel missing")

    print("QSOL-HARNESS contract self-test: OK")


if __name__ == "__main__":
    main()
