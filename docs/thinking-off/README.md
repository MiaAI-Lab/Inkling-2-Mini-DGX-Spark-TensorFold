# Direct answers with thinking off

Patch `0042-mini-thinking-off.patch` implements a direct-answer path for native
Inkling 2 Mini chat requests. It selects the checkpoint by `model_type` equal to
`inkling2_loop` and by its Inkling reply format, rather than the served model name
or the shared block format alone.

For a thinking-off request, the server supplies the initial answer or tool header,
uses effective reasoning effort `none`, and ends generation at the completed
answer/reply. This avoids generating a later thinking block rather than deleting
a reasoning buffer after generation. JSON and tool grammars account for the
prefilled header; later parallel tool calls retain their own model headers.
Whole and streamed replies start their parser in the corresponding state.

For Mini chat with thinking off, that choice takes priority over a conflicting
named effort and `ignore_eos:true`. Thus `ignore_eos:true` does not force a Mini
thinking-off chat to fill `max_tokens`; it may end once its answer is complete.
Thinking-on requests, raw completions and
other checkpoint types keep their existing paths and termination controls.
Weights, kernels, sampling and recipe defaults are unchanged.

Thinking-off quality can be lower, including incorrect answers and weaker
selection of an automatic tool call. Required/named tool choices retain their
constraints. Custom chat-template overrides are not qualified by these tests.

## Host regression

In an engine checkout with this recipe's patch set applied, use the pinned Zig
version and the matching grammar library:

```sh
TF_INKLING_TEST_DIR=/path/to/Inkling-2-Mini \
LD_LIBRARY_PATH=/path/to/grammar/lib \
zig build test-inkling-mini-host --summary all
```

The included tests cover actual request preparation, whole/streamed reply
parsing, effort and EOS precedence, unchanged on/raw controls, non-Mini metadata,
and real-tokenizer JSON/tool framing. Examples are independently written; no
application workloads or logs are needed. The real-checkpoint test skips when
`TF_INKLING_TEST_DIR` is unset. These are host integration checks, not GPU backend
qualification.
