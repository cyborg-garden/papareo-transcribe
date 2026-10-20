---
name: papareo-transcribe
description: Transcribing recordings people send you - voice notes and audio files - with transcribe_recording, asking first whether they want it local or through Papa Reo (te reo Māori), and what to say back. Read whenever someone sends a recording or asks for a transcript.
---

# Transcribing recordings

On this agent, a recording someone sends you is **something to transcribe**, not a message to answer. Voice input is switched off here, so a voice note arrives as a line like `[The user sent a voice message: /opt/data/... (duration: 0:42)]`, and an audio file arrives as a path.

## Ask first

Before transcribing, ask the person which way they want it, unless they've already said. One friendly question, nothing else:

> Happy to! Would you like a local transcript (free), or one from Papa Reo, which is best for te reo Māori? Papa Reo costs a little, and I won't see the text.

If your system instructions say you are deployed in line with the Kaitiakitanga License, drop "and I won't see the text":

> Happy to! Would you like a local transcript (free), or one from Papa Reo, which is best for te reo Māori (it costs a little)?

- **Don't comment on the file.** Voice notes often arrive as `.bin` or with no extension; that's normal and the tool handles it. Don't inspect, rename or describe the file.
- **One recording, one question.** Don't count or bring up earlier recordings unless the person does. If they send several at once, one question can cover them all.
- If they said it up front ("transcribe this with Papa Reo"), don't ask. Don't pick for them, and don't run both.

| They say something like | engine |
|---|---|
| "local", "just a normal transcript", "it's English" | `local` |
| "Papa Reo", "it's in te reo", "treat it as Māori" | `papareo` |

## The tool

`transcribe_recording(path, engine, chosen_by_person)`, with `engine` either `"local"` or `"papareo"`, and `chosen_by_person: true` only when the person told you which. Without it the tool does nothing and tells you to ask. Pass the path exactly as the message gave it, even if it ends in `.bin` or has no extension: the tool checks the content, so don't rename, copy or convert the file first. It **sends the transcript to this chat as a .txt file itself**. Don't send it again, and don't call `ovenmitt_vault_send` for it.

- **`local`**: whisper on this machine. You get back the `text`. Say it's attached and what made it (the result's `transcribed_by`, e.g. "whisper large-v3-turbo, run locally"). **Don't summarise, translate or quote it unless they ask.** You can answer questions about it when they do.
- **`papareo`**: the recording goes to Papa Reo, Te Hiku Media's speech recognition for te reo Māori. You get back a `handle`, and usually no text: the transcript is protected under the Kaitiakitanga License and must not pass through you. Say it's attached and that Papa Reo made it. Don't summarise it, guess at it, or ask to see it. See the `ovenmitt` skill for what a handle is.
- **`papareo` on an agent declared compliant**: if the result also has `text` and `"kaitiakitanga_compliant": true`, the people who run you have declared you deployed in line with the licence (your system instructions say so too). You may then work with that text **when the person asks**: answer questions, summarise, translate. Still say it's attached rather than repeating it unprompted, and never pass it to another tool or service (web search, other agents, files outside the vault, messages to other chats).
- A `note` in the result means something didn't go as planned, for example Papa Reo wasn't available. Pass it on in plain words. If it suggests a local transcript instead, ask before running one.

Each Papa Reo run costs money, so never retry it on your own; let the person ask.

## The Kaitiakitanga License

Papa Reo transcripts come under the [Kaitiakitanga License](https://papareo.io/kaitiakitanga), and you must follow it: the transcript doesn't pass through you, and you never pass it to another tool or service.

The transcript file ends with a footer that asks them to follow the licence's terms. It is important that they should also abide by the Kaitiakitanga License. You should say every time that it is a Papa Reo transcript, which will allow them to abide by the terms of the licence, but they will read the information in the footer of the transcript, so they don't need explicit reminding. Use your judgement: if they ask about the licence, or it looks like they are about to violate its terms, explain it to them and give the link.

## Resending

Papa Reo transcripts stay in the vault for a while. If someone asks for one again, use `ovenmitt_vault_list` and `ovenmitt_vault_send` (the label is the filename).
