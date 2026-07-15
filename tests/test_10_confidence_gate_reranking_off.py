"""Bug 10: when reranking is disabled (Config.ENABLE_RERANKING = False), the
reranker_agent must not crash, and must not fabricate a calibrated confidence
score from raw (non-cross-encoder) vector distances -- retrieval_confidence
should come back as None ("no signal"), and different underlying doc scores
must not collapse to some hardcoded constant.

Run in-process inside the live container via docker exec so we exercise the
real reranker_agent() code path without disrupting the shared container by
restarting it with an env var override."""
import subprocess

script = """
from backend.config import Config
Config.ENABLE_RERANKING = False

from backend.agents.reranker import reranker_agent, route_after_reranker

def make_state(scores):
    return {
        "query": "does it matter",
        "doc_results": [
            {"text": f"chunk {i}", "metadata": {}, "score": s, "source": "vector"}
            for i, s in enumerate(scores)
        ],
    }

# Two states with clearly different underlying vector distances.
state_a = make_state([0.1, 0.5, 0.9])
state_b = make_state([5.0, 6.0, 7.0])

out_a = reranker_agent(state_a)
out_b = reranker_agent(state_b)

print("CONFIDENCE_A=" + repr(out_a["retrieval_confidence"]))
print("CONFIDENCE_B=" + repr(out_b["retrieval_confidence"]))
print("RANKED_A_LEN=" + repr(len(out_a["ranked_results"])))
print("ROUTE_A=" + repr(route_after_reranker({**state_a, **out_a})))

# Empty doc_results must also not crash.
empty_out = reranker_agent({"query": "x", "doc_results": []})
print("EMPTY_CONFIDENCE=" + repr(empty_out["retrieval_confidence"]))
"""

result = subprocess.run(
    ["docker", "exec", "docurag_dev", "python3", "-c", script],
    capture_output=True, text=True, timeout=60,
)
assert result.returncode == 0, f"reranker_agent crashed with ENABLE_RERANKING=False: {result.stderr}"

out = {}
for line in result.stdout.splitlines():
    if "=" in line and line.split("=", 1)[0].isupper():
        k, v = line.split("=", 1)
        out[k] = v
print(f"  raw output: {out}")

confidence_a = eval(out["CONFIDENCE_A"])
confidence_b = eval(out["CONFIDENCE_B"])
ranked_a_len = eval(out["RANKED_A_LEN"])
route_a = eval(out["ROUTE_A"])
empty_confidence = eval(out["EMPTY_CONFIDENCE"])

# reranked=False (since ENABLE_RERANKING is off) means confidence should be
# None -- there's no calibrated signal, and the code explicitly avoids
# fabricating one from raw distances.
assert confidence_a is None, f"expected None confidence with reranking off, got {confidence_a}"
assert confidence_b is None, f"expected None confidence with reranking off, got {confidence_b}"
assert ranked_a_len == 3, f"expected all 3 doc_results passed through unranked, got {ranked_a_len}"
# route_after_reranker must gracefully route to synthesizer when confidence is None
# (not crash on a missing gate, not silently drop to web_search on no signal).
assert route_a == "synthesizer", f"expected None-confidence route to go straight to synthesizer, got {route_a!r}"
assert empty_confidence == 0.0, f"expected 0.0 confidence for empty doc_results, got {empty_confidence}"

print("PASS: bug 10 confidence gate reranking off (no crash, no fabricated hardcoded confidence, routes to synthesizer)")
