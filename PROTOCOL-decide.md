# Kohagi decide protocol (v1)

`kohagi-decide` is a pure function from id-tagged *typed requests* to id-tagged
*typed answers*, spoken over stdin/stdout as JSONL. A request is a state plus
named questions; an answer is, for each question, the probability of its
options and the `choice`, `score` or `noul` those imply. It is a separate
binary because it is a different function: `kohagi-rerank` takes `(query,
text)` pairs and gives one number each, while this takes a closed set of
candidates and gives a distribution over it.

Every envelope rule is the rerank protocol's — `id` is opaque and echoed, a
malformed line is skipped rather than fatal, a blank line means "answer what I
have sent and reply now", and a reply ends with a blank line of its own. This
document says where the records differ; see [PROTOCOL-rerank.md](PROTOCOL-rerank.md)
for the rules both share.

## Input (stdin, JSONL, UTF-8, one record per line)

```json
{"id": 1, "state": "二重請求です。返金してください。",
 "questions": {
   "department": {"type": "choice", "instructions": "担当部署は？",
                  "criteria": {"billing": "請求・返金", "technical": "技術的な障害", "sales": null}},
   "urgency": {"type": "score", "instructions": "対応の緊急度は？",
               "criteria": ["急がない", "早めに対応", "即時対応が必要"]},
   "refund": {"type": "noul", "instructions": "返金を要求しているか？"}}}
```

| field | type | notes |
|---|---|---|
| `id` | any JSON value | **Opaque.** Echoed verbatim; never interpreted. |
| `state` | string or JSON value | The situation. A non-string is serialized as compact JSON, as the reference does. |
| `questions` | object | One entry per named question; the keys come back as the answer names. |

Each question is an object:

| field | type | notes |
|---|---|---|
| `type` | `"choice"` \| `"score"` \| `"noul"` | Which shape to answer. |
| `instructions` | string | The question in words, nonempty. |
| `criteria` | object \| array | `choice`: `{label: description}`; `score`: ordered level strings; `noul`: omitted. |

- **Skips (non-fatal).** A line is skipped (exit code 2, with the reason on
  stderr) when it is not a JSON object, has no `id` or `state`, has an empty or
  non-object `questions`, or has a question that is malformed: no `type`, an
  unknown type, no `instructions`, a `choice` without a `criteria` object, or a
  `score` whose `criteria` is not an array of strings. Nothing is guessed.

## Output (stdout, JSONL, one record per line)

```json
{"id": 1, "answers": {
  "department": {"type": "choice", "choice": "billing", "probability": 0.998052,
                 "probabilities": {"billing": 0.998052, "technical": 0.000469, "sales": 0.001478}},
  "urgency": {"type": "score", "score": 1.265782, "probabilities": [0.169491, 0.395236, 0.435273]},
  "refund": {"type": "noul", "noul": 0.971384, "probabilities": {"true": 0.971384, "false": 0.028616}}}}
```

- `id` is the input value, unchanged. **Map by id, not by order.**
- `answers` is keyed by the input's question names. Each answer carries its
  `type` and the distribution the model produced:
  - **choice**: `choice` (the label key), `probability` (its probability), and
    `probabilities` (label → probability). The keys are the `criteria` keys.
  - **score**: `score` (the expected zero-based level, `Σ i·pᵢ`) and
    `probabilities` (one entry per level, in `criteria` order).
  - **noul**: `noul` (`P(true)`) and `probabilities` (`{"true": …, "false": …}`).
- Probabilities are rounded to six decimal places, so a run's output is stable.

## Heads

The request and the answer shapes above are the same whichever model answers.
What differs is the **head**, chosen with `--head` and independent of the
checkpoint chosen with `--model-id`:

- `cross-encoder` — a one-label cross-encoder scores `(context, candidate)`
  pairs one at a time, and their raw logits are softmaxed per question. This is
  the `argos1111` Jev-local shape (`--model-id
  argos1111/modernbert-ja-310m-jev`).
- `laya` — a packed encoder reads state, instructions and every option in one
  pass, and a decision head scores the marked options (the
  `convaiinnovations/laya` shape, English and multilingual). Behind the
  `head-laya` cargo feature, on by default; a build with `--no-default-features`
  drops it, and `--head laya` then fails with a message listing the heads it
  does have.

Heads are a registry (`src/decide/engine.rs`), so the set a build offers is
whatever its features compile in.

## How a question becomes a distribution

For the `cross-encoder` head, the candidates of one question are scored as
`(context, candidate)` pairs, and their raw logits are softmaxed over the
question. There is no other grouping: two questions, in one record or across
records, are independent.

The context and the candidate text follow the Jev-local prompt shape, so the
checkpoint reads what it was trained on:

- context: `質問: {instructions}\n状況: {state}`
- choice candidate: `{label} — {description}`, or `{label}` when the
  description is `null`
- score candidate: the level string itself
- noul candidate: `true` and `false`, in that order, so `noul` is the first
  probability

The rendering belongs to the head, not the caller: a checkpoint trained on a
different shape is a different `--head`.

## v1's contract

An answer carries only the probabilities and the value derived from them. The
decision model's escalation head (`act_probability`) is **not** returned: a
later version adds a field rather than changing these, so a caller reading the
three shapes above is unaffected.

## Exit codes

The same as the other binaries: `0` every line answered, `2` finished with
skipped lines, `1` fatal, `3` the requested CoreML backend cannot serve this
request.
