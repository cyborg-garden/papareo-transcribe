---
name: papareo-transcribe
description: Transcribing recordings people send you - voice notes and audio files - with transcribe_recording, asking first whether they want it local or through Papa Reo (te reo Māori), and what to say back. Read whenever someone sends a recording or asks for a transcript.
---

# Transcribing recordings

On this agent, a recording someone sends you is **something to transcribe**, not a message to answer. Voice input is switched off here, so a voice note arrives as a line like `[The user sent a voice message: /opt/data/... (duration: 0:42)]`, and an audio file arrives as a path.

## Ask first

Before transcribing, ask the person which way they want it, unless they've already said. Keep it to one short question, for example:

> Want a local transcript (free), or Papa Reo (best for te reo Māori; it costs a little, and I won't see the text)?

| They say something like | engine |
|---|---|
| "local", "just a normal transcript", "it's English" | `local` |
| "Papa Reo", "it's in te reo", "treat it as Māori" | `papareo` |

If they said it up front ("transcribe this with Papa Reo"), don't ask again. Don't pick for them, and don't run both.

## The tool

`transcribe_recording(path, engine)`, with `engine` either `"local"` or `"papareo"`. It **sends the transcript to this chat as a .txt file itself**. Don't send it again.

- **`local`**: whisper on this machine. You get back the `text`. Say it's attached and what made it (the result's `transcribed_by`, e.g. "whisper large-v3-turbo, run locally"). **Don't summarise, translate or quote it unless they ask.** You can answer questions about it when they do.
- **`papareo`**: the recording goes to Papa Reo, Te Hiku Media's speech recognition for te reo Māori. You get back a `handle` and no text. The transcript is protected under the Kaitiakitanga License: it must never pass through you. Say it's attached and that Papa Reo made it. Don't summarise it, guess at it, or ask to see it. See the `ovenmitt` skill for what a handle is.
- A `note` in the result means something didn't go as planned, for example Papa Reo wasn't available. Pass it on in plain words. If it suggests a local transcript instead, ask before running one.

Each Papa Reo run costs money, so never retry it on your own; let the person ask.

## Resending

Papa Reo transcripts stay in the vault for a while. If someone asks for one again, use `ovenmitt_vault_list` and `ovenmitt_vault_send` (the label is the filename).
