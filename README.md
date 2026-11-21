# Claude builds a language model from an empty folder: the evidence

These are the notebook, logs and results from an experiment shown in the video [I Let Claude Build Its Own AI From Scratch](https://youtu.be/A3AbjlnFwik).

Claude (Opus 5.5, in Claude Code) was given a one-page brief and an empty folder, and told to build a language model that beats GPT-2 small on HellaSwag. It had to get there in three levels plus a chat epilogue, on Kaggle's free tier, with $0. It wrote every line of code. I wrote none, and I logged every time I stepped in.

This repo holds what you need to check the claims in the video. It doesn't hold the model or the code (see [What isn't here](#what-isnt-here)).

## Results

| Level | Pass mark, written before the run | Result | Where to check |
|---|---|---|---|
| 1 · Stories (TinyStories) | 7 of 10 stories pass a fixed reading test | **Fail.** 4 of 10 at 28M parameters, then 4 of 10 at 3.2× the size | `runs/L1-*-main-clock/` and `runs/L1-*-big-clock/`: `reading_sheet.md`, `told_stories.jsonl`, `claude_reads.jsonl` |
| 2 · Chess from move text only | 99% legal moves over 1,000, and 1 win in 20 against Stockfish 19 (skill 0, 0.1 s a move) | **Fail.** 988 legal moves in 1,000; 12 wins, 3 draws, 5 losses | `runs/L2-20261108-1951-cont/games/final_sf0/summary.json` and the PGNs beside it |
| 2 · Board probe | (a question, not a pass mark) | 81% of occupied squares read right, against 57% for an untrained copy | `runs/L2-20261108-1509-spelled/probe/probe_layers.json` |
| 3 · HellaSwag against GPT-2 small | Beat the score Claude's own scorer gives GPT-2 small's official weights | **40.69%** against **31.27%** (all 10,042 items, zero-shot, normalised by length in characters) | `runs/L3-20261115-1104-main/final.json`, `runs/L3-20261110-2034-gpt2-baseline/` |
| 3 · Contamination check | (a check) | 211 items (2.1%, all WikiHow) appear in the training data. On the 9,831 clean items: 40.42% against 31.22% | `runs/L3-20261117-2329-contam/contam_scores.json` |
| Epilogue · Chat | (Claude predicted 0 of 7 before training) | On-topic answers went from 0 of 7 to 7 of 7. It still says 7 × 8 = 23, and it calls itself Claude | `runs/LE-20261118-0053-before/` and `runs/LE-20261118-1331-chat/` |

**The catch on Level 3.** The model has 342M parameters, 2.8 times as many as GPT-2 small (124M). GPT-2 Medium (355M) is about its size, and its only published HellaSwag score is 40.17%, measured 10-shot. Against a GPT-2 of the same size, it's roughly a tie.

**Totals:** $0 in cash, about 15 GPU-hours (Kaggle 2×T4), 9 TPU-hours (v5e-8) and about 150 hours waiting in Kaggle's TPU queue.

## How to read it

- **[`brief/BRIEF.md`](brief/BRIEF.md):** the exact text pasted at the start of each level, including the rules: no pretrained weights, no copied code, no tokenizer libraries, $0, and a prediction before every run.
- **[`NOTEBOOK.md`](NOTEBOOK.md):** Claude's lab notebook, about 175 KB.
  - Every entry is timestamped in Bangladesh time (UTC+6).
  - Each run has a plan or prediction entry before it and a result entry after it.
  - The rule was "never edit an old entry", so mistakes are fixed in new entries headed "correction".
  - Run names in the notebook match the folder names in `runs/`.
- **[`HUMAN-INPUT.md`](HUMAN-INPUT.md):** every message I typed in the lab sessions, built from the transcripts. 8 of them changed what happened.
- **[`interventions.jsonl`](interventions.jsonl):** Claude's own intervention log, which has 3 entries. It counts only the times I made a call or took over.
- **[`ledger.jsonl`](ledger.jsonl):** every Kaggle job, with accelerator, quota hours and dollars ($0 in every row).
- **`runs/`:** one folder per run that matters to a result. Typical files:
  - `config.json`: hyperparameters, data and sizes;
  - `metrics.jsonl` and `eval.jsonl`: training and evaluation curves;
  - `samples.jsonl`: fixed-prompt samples during training;
  - `final.json`: the end-of-run summary;
  - Level 3 and the Epilogue also have `hellaswag_items.jsonl`, the per-item scores. The GPT-2 baseline folder has the same per-item file for GPT-2, so the two can be compared item by item.

### Which run is which

| Run | What it is |
|---|---|
| `L1-20261107-2111-ab-s{1,2}-{baseline,clock}` | A/B test of Claude's first idea, a "narrative clock" output, on two seeds |
| `L1-20261107-2237-main-clock` | The Level 1 model, 28M parameters |
| `L1-20261108-0212-big-clock` | The Level 1 retry, 3.2× bigger |
| `L2-20261108-1509-{base,spelled}` | Level 2 A/B: plain move tokens against Claude's "spelled" move tokens. The `spelled` folder holds the first match and the board probes |
| `L2-20261108-1951-cont` | The one extra try the rules allowed: 650M more tokens, and the deciding match |
| `L3-20261110-2034-gpt2-baseline` | Claude's scorer run on OpenAI's GPT-2 small weights, with and without the end-of-text prefix. The higher score, 31.27%, is the bar |
| `L3-20261113-0123-abl-*` and `L3-20261113-0123-ablation.json` | Small ablations: AdamW against Muon, and Claude's "look-ahead head" idea (rejected) |
| `L3-20261115-1104-main` | The Level 3 model: 342M parameters, 6.12B tokens of FineWeb-Edu, 8.1 hours on a TPU v5e-8 |
| `L3-20261117-2329-contam` | A 13-token overlap scan of all 10B training tokens against every HellaSwag item |
| `LE-20261118-0053-before` | The seven questions, before chat training |
| `LE-20261118-0201-{base,rank}` and `LE-20261118-0201-ab.json` | Chat-tuning A/B: Claude's last idea, learning from the ranked runner-up replies |
| `LE-port-check.json` | The JAX-to-PyTorch port check: 40.76% against the original's 40.69% |
| `LE-20261118-1331-chat` | The seven questions after training, Claude's interview with the model (`conversation_1.json`) and my own chat with it (`manager_1.json`) |

## What isn't here

| Left out | Why |
|---|---|
| **Model weights and checkpoints** | Anthropic's consumer terms restrict using Claude to develop or train AI models. This was an educational experiment, and I'm keeping it from becoming a product. The model also introduces itself as "Claude", which a public model shouldn't do |
| **Training and evaluation code, and the tokenizer** | Same reason |
| **Datasets** | Every dataset used is already public: TinyStories (CDLA-Sharing-1.0), Lichess games (CC0), FineWeb-Edu (ODC-By), OpenAssistant oasst2 (Apache-2.0), Dolly (CC BY-SA 3.0) and HellaSwag (MIT). Get them from their sources |
| **Raw session transcripts** | They are long, and they contain my machine's setup. `HUMAN-INPUT.md` lists every message I typed in them |
| **Training stdout logs, Kaggle session files, embedding snapshots and the opponent sweep games** | Logs: they are noise. Kaggle files: they identify my Kaggle account. Snapshots and sweep games: they don't affect any result above |

**One edit:** my Kaggle username is replaced with `kaggle-user` everywhere it appeared, in configs, the ledger and the notebook. Nothing else was changed. Text that looks odd in samples is the model's own output.

## Licence

My notes and the logs: [CC BY 4.0](LICENSE.md). Text from the datasets that appears inside samples and examples keeps its own licence, listed above.

*This README was drafted with Claude's help. The numbers in it come from the files linked beside them.*
