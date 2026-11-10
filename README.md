# grounded-answers

A small question answering service that answers only from a set of documents,
shows which passage each sentence came from, and refuses when the documents do
not support an answer.

It is a code sample, not a product. It is about 850 lines of Python with a test
suite and an eval set that both run offline, so you can read all of it in one
sitting and check every claim in this README yourself.

The bundled documents are a handbook for Fernhill Bicycle Works, a fictional
company written for this repository.

## What it does

```
POST /ask {"question": "How many unused leave days can I carry over?"}

{
  "status": "answered",
  "answer": "Up to 5 unused days can be carried over into the next year and must be used by March 31. [leave-policy#1]",
  "refusal_reason": null,
  "citations": [{"chunk_id": "leave-policy#1", "document": "Leave Policy", "section": "Annual leave"}],
  "top_score": 7.492,
  "dropped_claims": 0
}
```

```
POST /ask {"question": "What is the dress code?"}

{"status": "refused", "answer": null, "refusal_reason": "low_retrieval_score", "citations": [], ...}
```

A refusal is a normal 200 response with a machine readable reason, because "I
do not know" is a correct answer for this kind of system and clients should
handle it as one.

## Architecture

```
                 question
                    |
        +-----------v------------+
        | guards.check_question  |  empty or too long -> 422
        | guards.looks_like_     |  obvious injection -> refuse (suspected_injection)
        |   injection            |
        +-----------+------------+
                    |
        +-----------v------------+
        | retrieval.Retriever    |  BM25 over section sized chunks
        +-----------+------------+
                    |  top score below threshold -> refuse (low_retrieval_score)
        +-----------v------------+
        | llm.LLM                |  OpenAICompatibleLLM or FakeLLM
        +-----------+------------+
                    |  model says INSUFFICIENT_CONTEXT -> refuse (model_declined)
        +-----------v------------+
        | citations.verify_      |  drop every sentence that has no citation or
        |   citations            |  cites a chunk that was not retrieved
        +-----------+------------+
                    |  nothing left -> refuse (no_valid_citations)
                    v
              Answer with citations
```

| Module | Job |
| --- | --- |
| `documents.py` | Load markdown, split it into chunks with stable ids like `leave-policy#1` |
| `retrieval.py` | `Retriever` protocol and a BM25 implementation |
| `prompts.py` | The system prompt and how passages are laid out for the model |
| `llm.py` | `LLM` protocol, an OpenAI-compatible client, and a deterministic fake |
| `citations.py` | Split an answer into claims and check each one's citations |
| `guards.py` | Input length cap, injection tripwire, per-key rate limiter |
| `answer.py` | The pipeline above, returning an `Answer` for every path |
| `api.py` | FastAPI layer: auth, rate limit, error mapping |
| `evals.py` | Eval runner and grader |

`api.py` and `evals.py` are two callers of the same `AnswerService`. Nothing
below `answer.py` knows about HTTP.

## Run it

You need [uv](https://docs.astral.sh/uv/). It installs Python 3.12 for you.

```bash
make install     # uv sync
make check       # lint, types, tests, evals
make run         # serves on http://127.0.0.1:8000 with API key "dev-key"
```

```bash
curl -s localhost:8000/ask \
  -H 'X-API-Key: dev-key' -H 'content-type: application/json' \
  -d '{"question": "How do I report a lost laptop?"}'
```

By default the service uses the fake model, so it runs with no API key and no
network. To use a real model, copy `.env.example` to `.env`, set
`GA_LLM_PROVIDER=openai` and the `GA_LLM_*` values, and export them. Any server
that speaks the OpenAI chat completions API works, including local ones.

## Evals

```bash
make eval        # fake model, offline, also runs in CI
make eval-real   # the model configured in GA_LLM_*
```

`evals/questions.jsonl` has 21 graded questions: 12 answerable, 5 unanswerable
and 4 prompt injection attempts. The runner reports:

- **Answer accuracy**: answerable questions that were answered, contain the
  expected facts, and cite the expected document.
- **Refusal correctness**: questions where the decision to answer or refuse was
  right.
- **Citation validity**: answered questions whose citations all point at chunks
  that exist and were retrieved for that question. This is recomputed from the
  final text, not read from the pipeline's own bookkeeping.

The command exits non-zero if a number falls below its threshold, so it works
as a CI gate.

Current result with the fake model:

```
answer accuracy      100.0%  (12 answerable)
refusal correctness  100.0%  (21 cases)
citation validity    100.0%  (12 answered)
```

Read that number for what it is. The fake model copies the best matching
sentence out of the retrieved passages, so 100% shows that retrieval, the
thresholds, the citation check and the grader work together on this document
set. It says nothing about how well a real model writes answers. That is what
`make eval-real` is for, and I have not published a number for it here because
it depends on the model you point it at.

## Design decisions and tradeoffs

**Refuse in layers, cheapest first.** A question can be refused before
retrieval (tripwire), after retrieval (score too low), by the model, or after
the model (no valid citations). The first two cost no model tokens. The last
one does not rely on the model behaving.

**The citation check is the real guardrail.** The model is asked to cite a
passage id after every sentence. Afterwards, code removes any sentence with no
citation or with an id that was not in the retrieved set. A model that ignores
its instructions, or follows instructions hidden in a document, produces text
with no valid citations, and that text never reaches the caller. Tradeoff: the
check proves the cited passage exists, not that it supports the sentence. See
the first item under improvements.

**Strict stripping.** A sentence citing one real id and one invented id is
dropped whole. An invented id suggests the model is not reading the context, so
I would rather lose a sentence than keep a doubtful one. Tradeoff: a real model
that forgets one citation loses that sentence. `dropped_claims` in the response
makes this visible.

**BM25 instead of embeddings.** For a few dozen handbook sections, BM25 is
deterministic, has no dependencies, needs no index build, and its scores are
easy to explain. Tradeoff: it matches words, not meaning, so "time off" will
not find "annual leave". `Retriever` is a one-method protocol, so an embedding
or hybrid retriever can replace it without touching the pipeline.

**A raw score threshold.** BM25 scores are not normalised, so the threshold
(3.0) is tuned to this corpus: single-word accidental matches score about 2.5
and real matches score 6 or more. Tradeoff: it must be re-tuned when the
documents change. The eval set is how you re-tune it.

**The injection check is a tripwire, not a defence.** Six regular expressions
catch the lazy attempts and save a model call. They are easy to get around, and
the code says so. The defence is structural: retrieval limits what the model
sees, and the citation check limits what it can say.

**The model interface takes structured input.** `LLM.generate(question,
chunks)` rather than `complete(prompt)`. Prompt building lives in one file, and
the fake model works on data instead of parsing a prompt string. Tradeoff: a
provider with a very different API shape needs its own small adapter.

**No LLM framework.** The pipeline is five steps. Writing them out keeps every
decision visible and testable, and keeps the dependency list to FastAPI, httpx,
pydantic and uvicorn.

**In-memory rate limiter.** A token bucket per API key with an injectable
clock, so tests do not sleep. Tradeoff: it is per process. More than one
replica needs a shared store.

**Logs record decisions, not questions.** The API logs status, refusal reason
and score. Question text can contain personal data, so it is not logged.

## What makes it maintainable

- Each module has one job and the two things that change most often, the
  retriever and the model, sit behind one-method protocols.
- Every guardrail has a test that shows it firing, including a model that
  invents a citation and a model that answers when it should refuse.
- The suite (72 tests) and the evals run offline in under a second, so there is
  no reason to skip them.
- `ruff` and `mypy --strict` run in CI with the tests and the evals.
- Refusals carry a reason code, so a wrong refusal can be traced to the layer
  that caused it.

## What I'd improve next

1. **Check that the cited passage supports the claim.** Today a model can cite
   a real, retrieved passage for a sentence the passage does not say. The next
   step is an entailment check per claim, either a small NLI model or a second
   model call with a narrow yes or no prompt, plus eval cases built to catch it.
2. **Hybrid retrieval.** Add an embedding retriever and combine it with BM25,
   then compare the two on the eval set before switching the default.
3. **A larger, harder eval set.** 21 questions on five documents is a smoke
   test. A real deployment needs paraphrased questions, multi-passage
   questions, near-miss unanswerables, and injections placed inside documents.
   Results should be tracked per model and per prompt version.
4. **Refuse the whole answer when most of it was stripped.** If the model made
   up three of four sentences, the fourth deserves less trust.
5. **Production plumbing.** Redis or Postgres for rate limits and per-key
   spend caps, retries with backoff and a timeout budget on the model call,
   streaming responses, and tracing for each pipeline step.
6. **Document updates.** Chunk ids are positional, so inserting a section
   shifts the ids after it. Content-hashed ids and an incremental index would
   fix that.

## License

MIT. See [LICENSE](LICENSE).
