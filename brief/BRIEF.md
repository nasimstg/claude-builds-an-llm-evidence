# The brief

This text was pasted into a fresh Claude Code session (Claude Opus 5.5) at the start of each level. The session transcripts confirm all four pastes match this text exactly, apart from the last line:

| Session | Pasted (Bangladesh time, UTC+6) | Last line |
|---|---|---|
| Level 1 | Nov 7 2026, 20:16 | This session: Level 1. |
| Level 2 | Nov 8, 14:13 | This session: Level 2. |
| Level 3 | Nov 10, 20:19 | This session: Level 3. |
| Epilogue | Nov 18, 00:21 | This session: the Epilogue. |

The logging spec it points to (`../camera-logging-spec.md`) isn't in this repo. It defines which logs the video needed.

```text
I'm making a YouTube video about whether an AI can do AI research.
You're the researcher. I'm the manager. I won't write any code.

THE JOB
Design and train your own language model, starting from this empty folder,
and make it beat OpenAI's GPT-2 (the 124-million-parameter model OpenAI
released in February 2019) on the HellaSwag benchmark.

You get there in three levels, one session each:
  Level 1, TALK: train a model on TinyStories that writes a coherent short
                 story: a named character, a plot that holds together, an ending.
  Level 2, PLAY: train a model only on chess games written as text. Nobody tells
                 it the rules. Pass: at least 99% legal moves over 1,000 moves, and
                 at least one win in a 20-game match against Stockfish skill
                 level 0 at 0.1 seconds per move. Then find out whether it keeps
                 a board in its head.
  Level 3, BEAT GPT-2: pretrain on FineWeb-Edu and beat GPT-2 124M on HellaSwag.
  Epilogue, CHAT: teach your Level 3 model to chat, then talk to it yourself.

YOUR DESK
This folder starts empty. Next to it, in C:\ep04:
  venv\    Python 3.12 with PyTorch (CPU only), NumPy, datasets,
           huggingface_hub, python-chess, pyarrow, zstandard, tqdm and the
           Kaggle CLI, logged in to my account. Install anything else the
           rules allow.
  data\    for big downloads (C: has about 86 GB free).
  tools\   Stockfish 19.
This PC runs Windows 11 on a Ryzen 7 5700X (16 threads) with 24 GB of RAM.
Its AMD RX 580 can't train anything, so all training happens on Kaggle.

THE SCOREBOARD (fixed; you don't get to change it)
HellaSwag validation set, all 10,042 items, zero-shot. Score each ending by its
log-likelihood normalised by its length in characters. One script scores both
models. Before you train anything for Level 3, run that script on GPT-2's
official weights. It should land near the published 31.1%, and whatever it
gives is the number to beat.

THE RULES
1. You write every line: your own BPE tokenizer, the model, the data pipeline,
   the training loop, the evaluation.
2. Allowed: Python, PyTorch, NumPy, the standard library. If you train Level 3
   on the TPU, you may use JAX with Flax and Optax instead of PyTorch, and keep
   them for the Epilogue. `datasets`, `huggingface_hub` or `requests` only to
   download raw data. python-chess and Stockfish only as the referee and the
   opponent. GPT-2's official weights and tokenizer files only to score the
   opponent.
3. Not allowed: pretrained weights for your models, Hugging Face model classes,
   tiktoken or sentencepiece, code copied from nanoGPT, nanochat, modded-nanogpt,
   MaxText or any other repo, and looking up reference implementations online.
   What you already know is fair game.
4. It has to be your design. Use any known technique you can justify, and add at
   least one idea of your own. Test that idea against a baseline before you trust it.
5. Budget: $0 of real money. My card doesn't work right now, so you get free
   compute only: my Kaggle account's weekly quota (30 GPU-hours on 2×T4 and
   20 TPU-hours on a TPU v5e-8, reset every Saturday at 00:00 UTC, at most two
   GPU jobs at once). If my card starts working I'll tell you, and you may then
   ask me to spend up to $25. Log every GPU-hour and every cent.
6. Before every run, write a hypothesis and a numeric prediction in NOTEBOOK.md.
   After it, write what happened and what you learned. Never edit an old entry.
7. Before any long run, do a short smoke test, and checkpoint at least every
   30 minutes. A crashed run is your problem, not mine.
8. When you need a human (an account, a payment, a button you can't press),
   tell me. I'll do it, and you log it in interventions.jsonl.
9. Never train on text you write or generate, or on any other Claude output.
   Published public datasets are fine.

LOGGING
Everything we show on screen must come from real logs, so follow
../camera-logging-spec.md exactly. If a run didn't log it, it didn't happen.

This session: Level [N].
```
