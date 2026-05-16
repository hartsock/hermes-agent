# hermes-agent ACP — headless dispatch gap

**Branch:** `drake-swarm/headless-dispatch-plan` (fork at
[`hartsock/hermes-agent`](https://github.com/hartsock/hermes-agent))
**Status:** Plan / scoping doc — no code changes yet.
**Audience:** drake-swarm (the multi-worker, multi-arbiter agentic
SDLC system being built in [`hartsock/gilabot`](https://github.com/hartsock/gilabot))
will turn this branch into an agentic project that produces a PR
back to upstream `NousResearch/hermes-agent`.

---

## TL;DR

We want **hermes-agent** to be one of the diverse LLM-family workers
in the drake-swarm bake-off (alongside `claude-code-acp` and
`codex-acp`). When we tried to wire it in during 2026-05-15, the
headless deployment story for `acp_adapter/` produced enough
friction that the work got parked. This TODO captures what we
hit, what we want to ship, and how drake-swarm will dogfood it.

---

## Why drake-swarm cares

Drake-swarm is a deliberately diverse-LLM agentic SDLC system. The
core insight: same-family LLMs reinforce each other's biases the
way same-background humans do, so the swarm runs bake-offs across
distinct model families and uses an arbiter quorum (≥3 distinct
families) to vote on candidate answers. See the workspace's
`DRAKE_SWARM_ASPIRATION.md`.

Today the drake worker pool has:

- `drake-pool-claude-worker` — Anthropic Claude via
  `@zed-industries/claude-code-acp` (LIVE)
- `drake-pool-codex-openai-worker` — OpenAI GPT via
  `@zed-industries/codex-acp` (parked: `~/.codex/auth.json` flow)
- `drake-pool-codex-dgx-worker` — DGX Ollama via codex-acp's OpenAI
  protocol shim (parked: `~/.codex/config.toml` + `OPENAI_BASE_URL`
  plumbing)
- **`drake-pool-hermes-worker` — Hermes-agent via `acp_adapter/`
  (parked: headless dispatch protocol gaps — see below)**

Once the headless story works, hermes becomes the **third distinct
family** (Hermes / Nemotron / Qwen on DGX Ollama) the swarm needs
for arbiter quorum, distinct from Claude (Anthropic) and Codex
(OpenAI).

---

## What we ran into (2026-05-15)

The drake-swarm tried to start `acp_adapter` as a long-running pod
that subscribes to a NATS subject, spawns a hermes-agent session
per task, and replies. Three concrete blockers surfaced:

### 1. LLM provider must be configured before ACP `initialize`

`acp_adapter/server.py` resolves the active LLM provider at process
startup. There's no fully-env-var-only path that goes from "fresh
container, only env vars set, no interactive `hermes setup`" to
"ready to handle `initialize`."

Symptoms we hit:

- `provider: "ollama"` (top-level) was rejected — schema actually
  expects `providers:` dict, not `provider:`.
- `custom_providers: [...]` (list) was rejected — schema expects
  `providers:` dict keyed by name.
- Even with `providers:` correct, the adapter wouldn't pick up an
  `OPENAI_BASE_URL`-style override at runtime without writing a
  `hermes.yaml` to disk.

drake-pool needs the worker to be **stateless** beyond the mounted
Secret + env vars. Pod restarts must produce identical state. A
"write hermes.yaml on container start, then exec the adapter" wrapper
works but is fragile — it duplicates schema knowledge into a shell
script.

**What we want**: env-var-only provider config. Specifically,
something like:

```bash
HERMES_PROVIDER_NAME=dgx-ollama
HERMES_PROVIDER_KIND=openai_compat
HERMES_PROVIDER_BASE_URL=http://dgx-ollama.messaging.svc.cluster.local:11434/v1
HERMES_PROVIDER_MODEL=hermes-3-llama-3.1-405b
HERMES_PROVIDER_API_KEY_ENV=DGX_OLLAMA_API_KEY  # optional, or skip
```

…and the adapter reads those at startup, no YAML required. The
existing YAML path should remain — env vars are a layer on top.

### 2. ACP protocol-version skew

Hermes ACP 0.9 / 0.10 emits a `usage_update` notification per
session. Rust ACP 0.11 (which `agent-client-protocol` 0.11.1 ships)
**removed `usage_update` from the schema entirely** — receiving it
causes the Rust client to error out before the first prompt response
makes it through.

We patched the fork (`fix/acp-suppress-usage-update`,
commit `47191497`) by adding an env knob
`HERMES_ACP_SUPPRESS_USAGE_UPDATE=1` that skips emitting the
notification.

**What we want upstream**:

- Either drop the `usage_update` emission entirely and rely on
  ACP 0.11+'s replacement (does it have one? — research item)
- Or version-gate the emission based on the negotiated protocol
  version: if the client said `protocol_version: "0.11"`, suppress
  it; if `"0.9"`/`"0.10"`, emit.

The current "suppress always" knob is a workaround, not the right
upstream shape.

### 3. Bootstrap scripts assume interactive setup

`acp_adapter/bootstrap/bootstrap_browser_tools.{ps1,sh}` and the
hermes-agent setup wizard generally are interactive: prompts for
provider, browser auth, etc. A drake-pool pod has no interactive
session — it spawns `python -m acp_adapter` and the adapter must
come up ready.

The acp_adapter already CAN run headlessly when everything's
pre-staged (we proved it locally after the YAML schema dance). What
it lacks is a documented "headless deployment recipe" — what env
vars to set, what files to mount, what's mandatory vs optional,
and a way to fail-fast with a clear error when something's missing
(today it fails late, deep in the LLM call path).

---

## Proposed contribution back to upstream

A series of small, atomic PRs to `NousResearch/hermes-agent`:

### PR 1 — Env-var provider configuration

Add a documented `HERMES_PROVIDER_*` env-var path that bootstraps a
single LLM provider entry without any YAML on disk. YAML continues
to take precedence if present; env vars are the fallback for
container-based deployments.

Acceptance:

- `docker run -e HERMES_PROVIDER_NAME=dgx-ollama … hermes-agent
  python -m acp_adapter --port 0` brings up an ACP adapter ready
  to accept `initialize` without any volume mounts.
- Unit tests covering env-var → provider record translation.
- `docs/deployment/headless.md` walking through the recipe.

### PR 2 — Protocol-version-gated `usage_update`

Replace the `HERMES_ACP_SUPPRESS_USAGE_UPDATE` env knob (from the
fork) with proper version gating: check the negotiated
`protocol_version` from `initialize`, suppress `usage_update` for
0.11+, emit for 0.9/0.10.

Or, if ACP 0.11+ has a replacement message for the same telemetry,
emit that instead. Research item before code.

Acceptance:

- Integration test that connects an ACP 0.11 client and confirms no
  `usage_update` is emitted, plus an ACP 0.10 client that DOES
  receive it.
- Removes the env-var workaround entirely once shipped — the fork's
  `fix/acp-suppress-usage-update` branch becomes obsolete.

### PR 3 — Headless fail-fast diagnostics

The adapter should detect missing config at startup, not at first
prompt. If `HERMES_PROVIDER_*` and `~/.config/hermes/config.yaml`
are both absent, exit non-zero with a single-line error like:

```
ERROR: No LLM provider configured. Set HERMES_PROVIDER_NAME etc.
or mount a hermes config at ~/.config/hermes/config.yaml. See
docs/deployment/headless.md for the headless deployment recipe.
```

Currently the adapter starts, accepts `initialize`, and only
errors out when the first prompt tries to call the LLM — which is
late enough that a worker pool dispatcher gives up.

### PR 4 (optional) — ACP worker mode

A new `python -m acp_adapter --worker` entry point that hard-codes
"this is a headless background worker, fail fast on misconfig, log
to stderr in JSON, expose a `/health` over stdio". Aligns with how
drake-pool / Zed expect to spawn ACP agents.

This may not be wanted upstream — could land just in the fork.
Research item.

---

## drake-swarm dogfood plan

This branch is intended to be **agentically built** by drake-swarm
once Path B (fan-out dispatch + arbiter quorum) is shipped.

The shape would look like:

1. `drake-swarm.toml` lives at the root of this fork with phases
   matching the four PRs above.
2. Each goal is small enough for a per-round bake-off:
   - "Add env-var parser in `acp_adapter/auth.py` that produces a
     `ProviderConfig` from `HERMES_PROVIDER_*`"
   - "Add fail-fast check at the top of `server.py:serve()`"
   - etc.
3. Workers: Claude (Anthropic), Codex (OpenAI), Hermes-on-DGX
   (when it works — chicken-and-egg only until PR 1 lands).
4. Arbiters: same three families, voting on each candidate.
5. The trace test is the existing pytest suite plus a minimal
   ACP smoke test (drake-pool-acp-worker can drive Hermes through
   one initialize/prompt cycle).
6. On consensus, drake-swarm pushes a feature branch + opens a PR
   to `NousResearch/hermes-agent`.

Until Path B lands, this TODO is the planning surface. Path A
(single Claude worker) could be used to ship PR 1 or PR 3 as a
warm-up exercise — they're each small enough.

---

## Bookkeeping

- **Drake-swarm context:** `gilabot/drake/drake-foreman/`
  (one-shot orchestrator) + `gilabot/drake/drake-pool/`
  (worker chassis). drake-pool's `deploy/k8s/hermes.yaml` is the
  parked manifest waiting on PR 1 + PR 2 to ship.
- **Tracking issues:**
  - gilabot#1662 — hermes worker, parked on the gaps above
  - gilabot#1663 — load hermes-agent into the drake-interactive
    image so we can dogfood it in the harness
  - gilabot#1664 — codex auth flow (sibling parked worker)
- **No code in this branch yet.** The first commit on this branch
  will land alongside the drake-swarm.toml that orchestrates the
  PRs above.
