---
name: papareo-transcribe
description: Transcribing recordings people send you - voice notes and audio files - with transcribe_recording, including when it escalates to Papa Reo for te reo Māori and what to say back. Read whenever someone sends a recording or asks for a transcript.
---

# Transcribing recordings

On this agent, a recording someone sends you is **something to transcribe**, not a message to answer. Voice input is switched off here, so a voice note arrives as a line like `[The user sent a voice message: /opt/data/... (duration: 0:42)]`, and an audio file arrives as a path. Either way, call `transcribe_recording` with that path.

## The tool

`transcribe_recording(path, engine="auto")` transcribes the recording and **sends the transcript to this chat as a .txt file itself**. Don't send it again.

- **English** (the usual case): you get back `"engine": "local"` and the `text`. Tell the person it's attached; you can answer questions about it.
- **May contain te reo Māori:** it goes to Papa Reo. You get back `"engine": "papareo"` and a `handle`, and no text. The transcript is protected under the Kaitiakitanga License: it must never pass through you. Say it's attached. Don't summarise it, guess at it, or ask to see it. See the `ovenmitt` skill for what a handle is.
- A `note` in the result means something didn't go as planned, for example the transcript is local because Papa Reo wasn't available. Pass the note on in plain words.

## When to pick an engine

Leave `engine` as `auto` unless the person tells you otherwise:

| They say something like | engine |
|---|---|
| "send this to Papa Reo", "it's in te reo", "treat it as Māori" | `papareo` |
| "just a normal transcript", "transcribe as NZ English" | `local` |

If an auto transcript came back local and they say it should have gone to Papa Reo, run it again with `engine="papareo"`. Each Papa Reo run costs money, so don't retry on your own; let the person ask.

## Resending

Papa Reo transcripts stay in the vault for a while. If someone asks for one again, use `ovenmitt_vault_list` and `ovenmitt_vault_send` (the label is the filename).
