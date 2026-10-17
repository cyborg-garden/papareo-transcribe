# papareo-transcribe

Transcripts of recordings, as a deliverable, for [Hermes](https://github.com/NousResearch/hermes-agent) agents. Someone sends the agent a voice note or an audio file and asks for a transcript; a `.txt` comes back in the same chat.

- **English:** transcribed locally with whisper, free. The agent also sees the text, so it can answer questions about it.
- **May contain te reo Māori:** the original audio goes to [Papa Reo](https://papareo.io), Te Hiku Media's speech recognition for te reo Māori and NZ English. The transcript goes straight into an [ovenmitt](https://github.com/utunga/oven-mitt) vault and arrives in the chat as an attachment. **It never enters the model's context**, because Papa Reo outputs are licensed under the [Kaitiakitanga License](https://github.com/TeHikuMedia/Kaitiakitanga-License/blob/tumu/papareo_api.md), whose clause 3 forbids sending them to services that may collect and use them. <!-- FIXME: move this utunga/ URL to cyborg-garden/oven-mitt as soon as that is fixed -->

The detector is the cost valve; the vault is the licence valve.

## How it works

```
transcribe_recording(path, engine="auto")
  1. local whisper (large-v3-turbo, language auto-detected)
  2. detector: whisper said "mi"?  any macron?  trigram share over threshold?
  3a. no  → .txt to this chat; the agent gets the text
  3b. yes → preflight: Papa Reo key, ovenmitt bridge, vault up, destination declared
            for this chat's platform; if any fails, 3a plus a note (nothing spent)
          → Papa Reo on the original audio → plain text + footer → vault → handle
          → ovenmitt_vault_send → this chat; the agent gets only the handle
```

`engine="papareo"` ("send this to Papa Reo", "it's in te reo") skips the detector; `engine="local"` ("just a normal transcript") skips escalation. Failure rules: a Papa Reo failure falls back to the local transcript with a note and is never retried automatically, because a retry spends money again; a failed vault write falls back the same way and says the call was paid for; a refused send leaves the transcript in the vault for `ovenmitt_vault_send` once the platform is allowed; a detector error escalates.

Papa Reo transcripts end with this footer, added before they enter the vault:

> This text transcribed by Papa Reo API. Please be sure to comply with the terms of the Kaitiakitanga license in use of this material.

## Layout

```
hermes-plugin/   the plugin: transcribe_recording          → plugins/papareo_transcribe/
hermes-skill/    SKILL.md                                  → skills/papareo-transcribe/
scripts/         fetch_corpus.py, train_detector.py, tune.py, papareo_smoke.py
tests/
```

Install both directories, each pinned to a tag (`git:cyborg-garden/papareo-transcribe#<tag>:hermes-plugin` and `…:hermes-skill`), alongside the ovenmitt bridge plugin and skill.

## What the agent needs

- The `ovenmitt_bridge` plugin and an ovenmitt daemon with a vault named `papareo`, holding a `send` destination named after each platform that may receive transcripts (`match = "<platform>/*"`, or one chat). Swarm Map's Transcription card adds the vault.
- Optionally, an `fs` destination named `folder` on that vault: every Papa Reo transcript is also exported there (best effort; a failed export doesn't stop the send).
- The `ovenmitt` Python package and ffmpeg in the image.
- `PAPAREO_API_KEY` in the gateway's environment, ideally as a compose `environment:` entry rather than in `.env`. The plugin pops it into memory at load. Without it, the agent transcribes locally only, and says so.
- `stt.enabled: false`, so voice notes reach the agent as recordings instead of being transcribed as messages; `stt.local.model: large-v3-turbo`; about 3 GB of memory.

## The detector

Three signals; any one escalates. It is biased to fire, because a false alarm costs a Papa Reo call while a miss costs transcript quality. The trigram signal is for mixed speech, where whisper picks English and drops macrons: it is the share of words whose character trigrams look more Māori than English, one number over the whole transcript.

The trigram model (`hermes-plugin/model/trigram.json`) is rebuilt by `scripts/fetch_corpus.py` and `scripts/train_detector.py` from:

- **te reo Māori:** [nga-tautohetohe-reo](https://github.com/TeHikuMedia/nga-tautohetohe-reo), the te reo Māori spoken in the New Zealand Parliament as recorded in Hansard, collected by Te Hiku Media. Public domain; ngā mihi for making it available. Utterances that are 70% or more reo are used as written and with macrons stripped.
- **English:** Katherine Mansfield, *The Garden Party* and *Bliss*, from Project Gutenberg. Public domain.

**The threshold is provisional** until it is tuned on real recordings. Put the recordings in a folder outside every repo, with a `labels.csv` (`file,should_escalate,kind,notes`), then:

```sh
scripts/tune-in-agent-image.sh <memos-dir>            # report
scripts/tune-in-agent-image.sh <memos-dir> --write    # and write the tuned threshold
```

The bar: every mixed and te reo recording trips. False positives on English are tolerated.

## The Papa Reo client

Built to the published API spec (`POST /tuhi/transcribe` under five minutes; the `/tuhi/transcribe/large` start-poll-download flow above that; `Authorization: Token <key>`). It is **switched off** (`LIVE_VERIFIED = False` in `hermes-plugin/papareo.py`) until one live call confirms it:

```sh
PAPAREO_API_KEY=... python -I scripts/papareo_smoke.py <a short recording of your own>
```

Not yet verified.

## Development

```sh
uv sync --extra dev && uv run pytest -q
```

Recordings, transcripts and the fetched corpus never go in git (see `.gitignore`).

Local transcription decodes audio with ffmpeg and hands whisper an array. faster-whisper 1.2.1's own decoder fails with PyAV 19 (`av.open() got an unexpected keyword argument 'metadata_errors'`).

## Licence

AGPL-3.0-only.
