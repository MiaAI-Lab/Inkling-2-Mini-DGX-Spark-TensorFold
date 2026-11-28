# Native thinking-budget handoff

Patch `0041-mini-thinking-budget.patch` registers the native Inkling block
transition with the recipe's pinned C4 server. This is a server change; it does
not change weights, CUDA kernels, sampling or recipe defaults.

With thinking enabled, an effective positive thinking budget closes the current
thinking block and opens an answer block. The complete transition is
`end_message` (200010), `message_model` (200001), then `content_text` (200004).
The model header is required for the native structured-output grammar to accept
the following answer. Generic `</think>` discovery cannot supply this transition.

Existing request/default budget selection remains unchanged. A request's positive
`thinking_budget` overrides the configured default; with no effective positive
budget, no budget cut is requested. Thinking-off requests do not receive a
thinking budget. Other reply formats retain their existing close detection.

## Host regression

In an engine checkout with this recipe's patch set applied, use the pinned Zig
version and the matching grammar library:

```sh
TF_INKLING_TEST_DIR=/path/to/Inkling-2-Mini \
LD_LIBRARY_PATH=/path/to/grammar/lib \
zig build test-inkling-budget-host --summary all
```

The test uses the public checkpoint's tokenizer and xgrammar with independently
written examples. It drives actual request preparation and lane budget handling,
checks the JSON answer after the forced close, and rejects an incomplete close
that omits the model header. It also checks default, zero and thinking-off budget
selection. Without `TF_INKLING_TEST_DIR`, the checkpoint-dependent test skips.
These are host integration checks, not GPU backend qualification.
