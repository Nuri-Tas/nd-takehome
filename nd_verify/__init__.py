"""Independent verifier for the ND take-home.

    from nd_verify import verify_text
    ok, reason, n_lines = verify_text("THM ( P > Q ) , P SEQ Q PRF N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N2 ; QED")

Operates on the whitespace-separated text format of spec.md; how you tokenise
internally is up to you, as long as your model's output decodes to that format.
This is the single source of truth for "valid proof" in the exam. Do not modify it;
if you believe it has a bug, report it in your write-up with a minimal example.
"""
from .verify import verify, verify_text, parse_formula, parse_proof_tokens, ParseError, RULE_NAMES
