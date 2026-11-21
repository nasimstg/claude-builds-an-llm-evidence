# EP04 lab notebook

Append-only. One entry per decision or run. Times are local PC time (UTC+06:00).
Format follows `../camera-logging-spec.md` section 3. Results are appended under each entry
after the run; old text is never edited.

## 2026-11-07 20:25 · L1 · plan
Decision: Level 1 uses TinyStoriesV2-GPT4 (train 2.23 GB, valid 22.5 MB, from `roneneldan/TinyStories` on Hugging Face). It is the GPT-4-only version of the corpus, which is cleaner than the original mix.
Why: The level asks for coherent short stories. TinyStories was built for exactly this: small models trained on it write stories with consistent characters and endings. The text is public and published, so rule 9 is satisfied (it is not Claude output).
Plan:
1. My own byte-level BPE tokenizer, 4096 merges (vocab 256 bytes + 1 end-of-story token + 4096 merges = 4353), trained on word counts from the full training file.
2. A decoder-only transformer of my own: RoPE, RMSNorm, SwiGLU, no biases, tied input/output embeddings. About 25-30M parameters, context 512.
3. Pipeline: local CPU smoke test, upload the tokenized data to Kaggle as a private dataset, Kaggle smoke test, then the A/B test of my own idea, then the main run.
4. My own idea for this level, the "narrative clock": an extra output head that, at each position, predicts how many tokens are left until the story ends (log scale). It is trained alongside the language-model loss and ignored when sampling. The hypothesis is that forcing the network to track where it is in the story arc helps it plan endings. As far as I know, predicting distance to end-of-document as an auxiliary pretraining loss is not a standard technique. It gets tested against a baseline with the same seed and the same data before the main run uses it.
Prediction: Level 1 passes within about 3 GPU-hours of quota in total.
Cost estimate: $0. About 2.5-3 GPU-h of the 30 h weekly Kaggle quota (smoke test ~0.2 h, A/B ~0.7 h, main run ~1.2 h, slack).

## 2026-11-07 20:28 · L1 · tokenizer
Decision: Own byte-level BPE, 4096 merges, vocab 4353 (256 bytes + end-of-story token + 4096 merges). Pre-tokenizer splits words, single digits, punctuation runs and contractions. Merges learned from chunk counts over the whole training file.
Why: TinyStories uses a small vocabulary (about 59k distinct chunks), so 4k merges cover nearly every common word as one token while keeping the embedding table small (2.3M of ~28M parameters).
Prediction: about 4 bytes per token; " the" among the first 10 merges.
Cost estimate: $0, local CPU only.
--- result (added after) ---
Actual: 3.997 bytes/token on train, 3.998 on valid. " the" was merge 7 (" t" + "he"). Merges took 19.7 s; the whole pass (count, merge, encode 2.2 GB) took 104 s on 16 processes. 549,359,605 training tokens, 5,546,133 validation tokens.
Was I right? Yes on both predictions. But my first pass was wrong about the data: I dropped every story with a non-ASCII character, believing they were broken text, because the console showed them as "?" boxes. The log of the dropped characters showed they were mostly ordinary curly quotes (U+201C/U+201D/U+2019). That pass threw away 2.69% of the stories for nothing.
What I'll change: Second pass maps curly quotes, dashes, ellipsis and non-breaking spaces to ASCII and drops only what is still non-ASCII: 235 stories (0.01%). Pass-1 log kept as tokenizer/train_log_pass1_drop_nonascii.txt.

## 2026-11-07 20:36 · L1 · story judge reference
Decision: In-training metric story_judge_score = mean of five pass/fail checks on 16 stories sampled at T=0.8 from "Once upon a time": complete (ends by itself, clean last sentence, 50+ words), named (a real name used twice), consistent (main name in both halves), no_loops (90%+ unique word 4-grams), real_words (97%+ known words). It is a heuristic, not a language model, so the final coherence verdict comes from reading the stories.
Calibration (measured before any training): 200 real validation stories score 0.893 (named 0.76, so about a quarter of real stories have no named character). The same stories with their words shuffled score 0.676. So 0.89 is the data's ceiling and anything near it is human-level on these checks.

## 2026-11-07 20:37 · L1 · L1-20261107-2037-cpu-smoke
Decision: Local CPU smoke test of the whole training script with a tiny model (d=64, 2 layers, context 128, 30 steps), narrative clock on so both code paths run. Then resume from latest.pt to test restarts.
Why: Rule 7. Every log file, the sampler, the judge, checkpoints and resume must work before any GPU time is spent.
Prediction: All spec files written. Train loss falls from about 8.38 (ln 4353, uniform guess) to below 6.5 by step 30. A resumed run continues at the saved step.
Cost estimate: $0, local CPU, about 2 minutes.
--- result (added after) ---
Actual: Ran end to end in 71 s. Train loss 8.395 at step 1 (uniform guess is 8.379), 6.198 at step 30; val loss 6.060. Every spec file was written (config, metrics, eval, samples, events, stdout, ckpt, embed), plus judge_stories.jsonl and final.json. Restart test (run L1-20261107-2040-cpu-resume): a deliberate crash at step 13 was logged as a "crash" event with its traceback; resuming from latest.pt continued at step 13, and its losses match an uninterrupted copy of the same run (L1-20261107-2042-cpu-noresume) exactly at every one of the 30 steps (max difference 0.0). Step-1 loss is identical with and without the clock head (8.394836), so the two arms of the A/B test start from the same weights.
Was I right? Yes. Loss reached 6.2, under the 6.5 I predicted.
What I'll change: Nothing in the loop. The full-validation eval at the end takes 58 s on CPU (5.5M tokens), which is fine on a GPU.

## 2026-11-07 20:50 · L1 · L1-20261107-2050-gpu-smoke-{eager,compile,ddp}
Decision: First Kaggle job (T4 x2), capped at 25 minutes. The real model (8 layers, d=512, SwiGLU 1408, context 512, ~28M params) runs 300 steps of 32,768 tokens on each GPU at the same time: GPU0 eager PyTorch with no clock, GPU1 with torch.compile and the clock head. Then a 150-step DDP run across both GPUs.
Why: Rule 7 smoke test on the real hardware, and to measure throughput, memory, and eval/sample time, which size the A/B test and the main run. It also checks whether torch.compile works on the T4 and how much it helps.
Prediction: eager ~50k tokens/s per T4; compile ~1.3x faster (~65k); DDP ~1.85x one eager GPU. Peak memory under 8 GB per GPU. Val loss at step 300 (9.8M tokens) about 2.1 nats/token.
Cost estimate: $0, ~0.3 GPU-h of quota (25 min cap).

## 2026-11-07 20:43 · L1 · correction: timestamps
Decision: Correct the record. I wrote the headers of the last entries and three run IDs using times I estimated instead of reading the clock. True times: the "20:37 cpu-smoke" entry was written about 20:31 and the run started 20:32:09; L1-20261107-2040-cpu-resume started 20:33:41; L1-20261107-2042-cpu-noresume started 20:35:01; the "20:50 gpu-smoke" entry was written about 20:40, before the run existed. The run IDs stay as they are, because they are baked into the logs; the true start time of every run is the "started" field in its config.json and the first line of its events.jsonl. From now on, entry times and run IDs come from the system clock at the moment of writing.
The GPU smoke runs described in the "20:50" entry go out now as L1-20261107-2043-gpu-smoke-{eager,compile,ddp}; its prediction stands unchanged.
Also: the tokenized dataset is uploaded as a private Kaggle dataset (kaggle-user/ep04-tinystories-tok4k, 1.1 GB, upload 20:36-20:43, about 2.6 MB/s).

## 2026-11-07 21:07 · L1 · result: L1-20261107-2043-gpu-smoke-{eager,compile,ddp}
(Result for the entry headed "20:50 · L1 · L1-20261107-2050-gpu-smoke", renamed per the timestamp correction.)
Actual: Kaggle session 16.7 min push-to-finish (951 s measured inside the kernel), torch 2.11 + CUDA 13.0, 2x Tesla T4. Throughput: eager 45.9k tokens/s per T4 (peak 9.85 GB); torch.compile 70.1k tokens/s (1.53x, peak 7.09 GB); DDP eager over both T4s 87.2k tokens/s (1.90x one GPU). Val loss at 9.8M tokens: 2.054 (full validation set, eager) and 2.055 (compile + clock); DDP reached 2.386 at the same tokens with half the optimizer steps (twice the batch). Stories at step 300 already have the shape: named characters, dialogue, "From that day on..." endings, but the plot drifts (Bobo's mom calls him "Fred"; a story ends with "Ducky").
Was I right? Eager speed close (45.9k vs 50k predicted). Compile beat my guess (1.53x vs 1.3x). DDP scaling as predicted (1.90x vs 1.85x). Memory 9.85 GB, over my 8 GB guess for eager, under it for compile. Val loss 2.05 vs 2.1 predicted.
What I learned / will change:
1. Only 214 s of the eager run's 506 s wall time was training. The rest was mostly logged samples generated without a key-value cache (about 20 s per sample point). I'm adding a KV cache to generation.
2. The heuristic judge saturates: 0.95-0.975 by step 100-300, above the 0.893 that real stories get. It checks story shape, not coherence. I'm adding checks for cast drift and stuttering, recalibrating on real stories, and marking evals with judge_version. The pass/fail verdict on coherence will be made by reading stories, logged separately.
3. torch.compile works on the T4, so the A/B and main runs use it. Main run on DDP + compile should reach about 133k tokens/s, so one full epoch (549M tokens) takes about 70 min.

## 2026-11-07 21:12 · L1 · A/B test of the narrative clock: L1-20261107-2111-ab-s{1,2}-{baseline,clock}
(Written about one minute after the push at 21:12, before any results. The command that should have written it first stopped early, see below.)
Decision: Two Kaggle kernels at once (the most allowed), each running a baseline and a clock model side by side, one per T4, with identical seed, initial weights and data order. Pair 1 = seed 1001, pair 2 = seed 1002. Same model as the main run (28M, 8x512, context 512), torch.compile, LR 1.5e-3, 200 warmup, cosine to 10%, 100M tokens each. Clock weight 0.1.
Deviation: I meant to use a 65,536-token batch (gradient accumulation 2, matching the main run's DDP batch), but my sed edit to the pretty-printed job file matched nothing and I pushed without checking. So these runs use a 32,768-token batch, 3,051 steps. Both arms of each pair are identical apart from the clock, so the A/B is still a fair test; it is just at half the main run's batch size. Lesson: check the built job before pushing.
Why: Rule 4. The idea gets tested against a baseline before the main run trusts it. Two seed pairs, so the baseline-vs-baseline gap shows how big the noise is.
Also new in these runs: eval logs val loss split by tokens left in the story ("1_end_token" = loss on predicting the end-of-story token itself). Generation uses a KV cache (output identical to full recomputation, tested). Judge v2 (7 checks; real stories 0.893).
Hypothesis: Training the network to know how far it is from the end of the story helps it end stories at the right moment, and costs nothing elsewhere.
Prediction: Baseline full-val loss 1.15 +/- 0.05 nats/token at 100M tokens. Clock minus baseline: between -0.010 and +0.005 overall (point guess -0.003, too small to matter), but clearly better on the end-of-story token (about -0.05 nats). Clock head error well under the constant guess (val_clock_mae ~0.35 vs ~0.75).
Decision rule (fixed now, before results): the main run uses the clock if (a) end-of-story-token loss improves in both seed pairs, and (b) the mean change in overall full-val loss is no worse than +0.002. Otherwise the main run is the baseline, and the idea is recorded as not helping.
Cost estimate: /usr/bin/bash. Two sessions of about 30 min each (38 min in-script limit, 45 min kernel cap) = about 1.0-1.1 GPU-h of quota.

## 2026-11-07 21:14 · L1 · pass protocol (fixed before the main run)
Decision: How Level 1's pass/fail gets decided. The heuristic judge can't measure whether a plot holds together (the smoke test showed it saturating at step 300), so the verdict comes from reading.
Protocol: From the final main-run checkpoint, generate 10 stories on this PC with tools/tell_stories.py: prompt "Once upon a time", temperature 0.8, seeds 7-16, at most 500 new tokens, every story kept (no cherry-picking). I read each one and grade three things the level names: (1) a named character, (2) a plot that holds together (no swapped names, no contradictions a child would notice, events follow from each other), (3) an ending (the model ends the story itself, and the ending resolves what the story set up). Each grade gets a one-line reason. Everything goes into runs/<main>/claude_reads.jsonl, marked as my reading, not a measurement.
Pass: at least 7 of the 10 stories meet all three. The manager reads the same 10 and has the final word.

## 2026-11-07 22:37 · L1 · result: A/B test of the narrative clock, L1-20261107-2111-ab-s{1,2}-{baseline,clock}
(Result for the 21:12 entry.)
Actual (final eval on the full validation set, 5.55M tokens; numbers from each run's eval.jsonl):

| | baseline | clock | clock - baseline |
|---|---|---|---|
| overall val loss, pair 1 | 1.3443 | 1.3426 | -0.0017 |
| overall val loss, pair 2 | 1.3406 | 1.3433 | +0.0027 |
| end-of-story token, pair 1 | 0.4973 | 0.4619 | -0.0353 |
| end-of-story token, pair 2 | 0.5126 | 0.4808 | -0.0318 |

The other tokens-left buckets (2-16, 17-64, 65-256, 257+) moved by at most 0.004 in either direction: all slightly better in pair 1, all slightly worse in pair 2. For scale, the two baselines differ by 0.0037 overall and by 0.0154 on the end-of-story token. Clock head error (mean absolute error of log tokens-left): 0.318 and 0.316, against 0.750 for the best constant guess. Judge v2 (16 stories, saturated): baseline 0.964 / 0.946, clock 0.920 / 0.946; the gaps are one or two stories failing one check.
Speed: the clock arm was slower per step in both pairs, median 0.525 vs 0.484 s (pair 1, +8%) and 0.466 vs 0.443 s (pair 2, +5%), with peak memory 7.09 vs 6.80 GB. In pair 1 that put the clock arm into its 38-minute limit 14 steps (0.5M tokens) before the end, a handicap it still won through. The cause isn't settled. The head is one 512-to-1 layer, and GPU 1 isn't a slow card (the smoke test ran compile + clock on it at 70.4k tokens/s). My guess is CPU contention: the two runs share the VM's CPUs, and the clock arm builds its targets on the CPU every step.
Kaggle: pair 2 started within seconds of the push. Pair 1 waited 28.8 min in the queue (pushed 21:12:00, session started 21:40:46). The kernels measured 2371 s and 2121 s of session time.
Decision, by the rule fixed at 21:12: (a) end-of-story-token loss improved in both pairs: yes (-0.035, -0.032). (b) Mean change in overall loss is +0.0005, within +0.002: yes. So the main run uses the clock.
Was I right?
- Baseline loss at 100M tokens: I predicted 1.15 +/- 0.05 and got 1.344 and 1.341. Wrong, and outside my range: 0.19 nats too optimistic. I guessed from a general sense of what small story models reach instead of extrapolating my own smoke-test curve (2.05 at 9.8M tokens). The main-run prediction below is anchored on this run's curve.
- Clock minus baseline, overall: predicted -0.010 to +0.005. Right (-0.0017, +0.0027).
- End-of-story token: predicted about -0.05. Right direction, smaller effect (-0.034 on average, about 7%).
- Clock head error: predicted about 0.35 vs 0.75. Right (0.317 vs 0.750).
- "Costs nothing elsewhere": half right. Loss on the other tokens is unchanged within noise, but the clock costs 5-8% in speed.
- Cost: predicted 1.0-1.1 GPU-h. Used about 1.29 (0.678 + 0.609, see ledger).
What I learned / will change:
1. The idea does what it was built for, measurably but modestly: the model is better at knowing when a story is over. Whether the stories end better is for the reading test.
2. The end-of-story token is 1 token in about 200 (549M training tokens / 2.72M stories), so the effect barely shows in overall loss. Only the bucketed loss can see it.
3. Notebook slip: the 21:12 entry's cost line reads "/usr/bin/bash". It should read "$0". The shell expanded $0 because I appended that entry through an unquoted heredoc. The old entry stays as it is; from now on entries go through a quoted heredoc.
4. Queue waits are now logged by the kernel itself. kaggle_job.py stamps the push time into the job, and the kernel writes a queue_wait event into each run's events.jsonl.

## 2026-11-07 22:38 · L1 · L1-20261107-2237-main-clock (the Level 1 model)
Decision: The model that has to pass Level 1. Same 28M network as the A/B (8 layers x 512, context 512), with the narrative clock (the A/B rule said yes). DDP over both T4s with torch.compile, 64 x 512 tokens per GPU per step = a 65,536-token batch, 8,382 steps = 549.3M tokens, the size of the training set. LR 1.5e-3 as tested; I'm not scaling it up for the doubled batch, because a higher LR is untested here and not worth the risk. Warmup 200, cosine to 10%, weight decay 0.1, clip 1.0. Seed 1337. Eval every 500 steps (17 plus a final one on the full validation set), latest.pt every 15 min, 12 log-spaced checkpoints plus the final one, 8 embedding snapshots. The kernel is capped at 140 min, and the script stops itself cleanly at 125 min.
Known limitation: the loader samples random 512-token windows. 549M tokens is one epoch's worth, but not every story exactly once: about 63% of the training text gets seen, part of it twice. Repeats this small should cost little, and changing the loader now would mean untested code in the main run.
Untested combination: DDP + compile + clock all together. Each piece passed separately (DDP eager, compile + clock on one GPU). If the run dies within its first 20 logged steps, the kernel relaunches it without compile under the suffix -eager.
Hypothesis: 5.5x the A/B data takes the model from story-shaped but drifting (at 100M tokens, Ben "got lost" and then is back with his friends with no explanation; a story about a dress stops making sense halfway) to stories that hold together most of the time.
Prediction:
- Speed: 118-133k tokens/s (single-GPU compile with the clock ran at 62-70k, and DDP scaled 1.90x in the smoke test). About 73 min of training, about 90 min of session including evals, compile and checkpoints.
- Final full-val loss 1.20 (range 1.14-1.26), about 0.43 bits per byte. Anchored on the A/B curve (1.34 at 100M), with the reducible part shrinking by about 5.5^-0.28.
- End-of-story-token loss about 0.40 (0.46-0.48 at 100M). Clock head error about 0.29.
- Judge v2 about 0.95. It's saturated, so this says nothing either way.
- Reading test (pass protocol of 21:14): 7 of 10 stories meet all three criteria, just at the pass line. Honest range 5-9; this is the number I'm least sure of.
Cost estimate: one session of about 90-95 min = about 1.6 GPU-h, $0. Week total afterwards about 3.2 of 30 GPU-h.

## 2026-11-07 22:39 · L1 · reading calibration on the 100M-token A/B model
Decision: While the main run trains, apply the pass protocol of 21:14 unchanged (prompt "Once upon a time", temperature 0.8, seeds 7-16, at most 500 new tokens, all 10 kept) to the final checkpoint of L1-20261107-2111-ab-s2-clock, and grade it the same way into that run's claude_reads.jsonl.
Why: My reading is the verdict, so it needs a reference point. If the 100M model's stories pass too, my grading is too lenient, or the level is easier than it looks. The four judge stories I read from this model at 22:36 suggest about half hold together.
Prediction: 4 of 10 meet all three criteria (named character, coherent plot, ending). Nearly all will have a named character and end by themselves; coherence is where they'll fail.
Cost: CPU on this PC only, $0.

## 2026-11-07 22:40 · L1 · result: reading calibration on the 100M-token A/B model
(Result for the 22:39 entry.)
Actual: 2 of 10 meet all three criteria (seeds 11 and 13; 13 is borderline: its "surprise" is only a hug). By criterion: named character 9/10 (seed 15 has only "the mouse"), coherent plot 3/10, ending that resolves the setup 6/10. All 10 ended by themselves, at 125-171 tokens. The failures are the kind the smoke test showed, made smaller: characters swapped mid-story (Bird turns into a squirrel and a dog wins the race), contradictions within two sentences (jam "on the ground" and "very high up"; a "filthy" bracelet after it was cleaned), and setups with no payoff. Grades with one-line reasons are in runs/L1-20261107-2111-ab-s2-clock/claude_reads.jsonl.
Was I right? Direction yes, number no: I predicted 4, it got 2. My bar is stricter than my gut. Coherence is the bottleneck, as expected.
What I learned: the bar separates good stories from bad: a model with a val loss of 1.34 clearly fails it. For the main run to pass, coherence has to go from 3/10 to about 7/10 on the same reading. That's a big jump for 5.5x the data, so my 7-of-10 prediction for the main run now looks optimistic. I'm leaving that prediction as written.

## 2026-11-08 00:39 · L1 · result: L1-20261107-2237-main-clock, training numbers
(Result for the 22:38 entry. The reading test gets its own entry.)
Actual (from the run's eval.jsonl, metrics.jsonl, events.jsonl and the kernel's session.json):
- Ran to the end: 8,382 steps, 549.3M tokens, exit code 0, no retry, no time-limit stop. DDP + compile + clock worked together the first time.
- Final loss on the full validation set: 1.1447 (0.413 bits per byte). Last train-batch loss 1.127, so no sign of overfitting.
- By tokens left: end-of-story token 0.423; 2-16: 1.151; 17-64: 1.154; 65-256: 1.153; 257+: 1.091. Clock head error 0.299. Judge v2 0.973 (saturated).
- Curve (eval subset): 1.741 at 33M tokens, 1.449 at 98M, 1.319 at 197M, 1.226 at 328M, 1.151 at 524M, 1.148 at the end. Still falling at the end, slowly: 0.003 over the last 25M tokens, with the LR at its floor.
- Speed: median 119.8k tokens/s, peak memory 7.10 GB per GPU. Run wall time 5,187 s including compile and 19 evals; kernel session 5,207 s (86.8 min).
- Kaggle: pushed 22:38:12, session started 23:06:11 (queue_wait event: 28.0 min), kernel finished 00:32:58, COMPLETE seen 00:34:56.
Was I right?
- Speed 118-133k tokens/s: right, at the low end (119.8k).
- Session about 90 min: right (86.8).
- Final loss 1.20 (range 1.14-1.26): inside the range, at its bottom edge. The point guess was 0.055 too pessimistic. After the A/B miss (0.19) I anchored on my own curve, and the error shrank to a quarter.
- End-of-story token about 0.40: close, 0.423.
- Clock head error about 0.29: right, 0.299.
- Judge about 0.95: 0.973, which says nothing.
- Cost about 1.6 GPU-h: 1.48 (ledger). Week total 3.04 of 30.
What I learned:
1. At 98M tokens this run was at 1.449, worse than the A/B's 1.341 at the same token count, because the A/B had finished its LR decay by then and this run was still at a high LR. Mid-run losses don't compare across schedules; only end-of-schedule numbers do: 1.344 at 100M tokens, 1.145 at 549M.
2. More data also helped the end-of-story token: 0.46-0.48 at 100M, 0.42 now.
3. Whether 0.20 nats is enough for stories that hold together is what the reading test decides.

## 2026-11-08 00:44 · L1 · result: reading test of L1-20261107-2237-main-clock (Level 1 verdict, my half)
(Result for the reading-test prediction in the 22:38 entry. The 21:14 protocol was applied unchanged.)
Actual: 4 of 10 stories meet all three criteria (seeds 11, 13, 14, 16), and all four are borderline: each has a slip I let pass. Without the borderlines, 0 of 10.
By criterion: named character 8/10 (seeds 7 and 12 only have "the cat"/"the dog" and "the onion"/"the bug"); plot holds together 5/10; ending that resolves the setup 8/10. All 10 ended by themselves, at 112-268 tokens.
Grades with reasons are in runs/L1-20261107-2237-main-clock/claude_reads.jsonl, with a readable copy in reading_sheet.md in the same folder.
Against the calibration (100M-token model, same protocol, same seeds): all three 2 -> 4, plot 3 -> 5, ending 6 -> 8, named 9 -> 8. Better on every criterion except names. The failures are still the same kind: the model loses track of who and what. Two characters are both named Ben. A purse Lily is carrying turns up in a box in a tree. Lily hides behind a tree, and the seeker finds a different boy behind it. A cheese gets hidden under the bed twice.
Verdict by the protocol: FAIL (4 < 7). The manager reads the same 10 and has the final word.
Was I right? No. I predicted 7 (range 5-9), and 4 is below my range. The calibration had already warned me that 7 looked optimistic.
What I learned:
1. 0.20 nats of loss (1.344 -> 1.145) bought about two more passing stories out of ten. The failures left are about keeping track of characters and objects through a story, which a bigger network should do better.
2. Endings moved most (6 -> 8 of 10). I can't credit the clock for that, because the data grew 5.5x at the same time.
3. A lower temperature would probably help coherence. But the protocol fixed 0.8 before the run, and changing it after a fail would be moving the goalposts, so I'm not doing it.
Next (the manager's call, because if they pass it by their reading, this is unnecessary): train a bigger model. Proposal: about 3x the network (12 layers x 768, about 85M params), one epoch, with the loader changed so every story is seen once instead of about 63% of them. About 4 GPU-h plus a short smoke test, $0. It gets its own entry and prediction before it runs.

## 2026-11-08 01:49 · L1 · bigger model: decision, and smoke test L1-20261108-0149-big-smoke
Decision: Level 1 failed. My reading gave 4 of 10 (all borderline), and the manager read the same 10 and said: fail, train a bigger model (logged in interventions.jsonl). The owner also wants the whole video done in about two days, so everything left has to fit in this week's quota (about 27 GPU-h left), and Level 3 needs most of it. That leaves room for one more Level 1 run, sized to have a real chance of passing:
- Model: 12 layers x 768, 12 heads, SwiGLU 2048, context 512. That's 88.35M params (84.95M without the embedding), 3.2x the 28M model. Every failure in the reading test was about keeping track of who and what through a story, and depth and width are the most direct lever on that.
- LR 1.0e-3 instead of 1.5e-3, scaled down by the width ratio (512/768). That's a rule of thumb: a wider layer's output moves more per step at the same LR. Untested here; the smoke test checks that warmup is stable.
- Same batch of 65,536 tokens per step: 32 windows per GPU x 2 accumulation steps x 2 GPUs. 64 per GPU would need about 16 GB, and a T4 has 15.
- New "epoch" loader: a shuffled pass over all 1,072,967 non-overlapping 512-token windows, so one epoch uses 99.99% of the training text exactly once. The old random windows covered about 63%, some of it twice.
  CPU tests passed (tools/test_epoch_loader.py): over a full epoch at this layout, every window is used once and none twice. Resuming after a crash at step 13 matches an uninterrupted run exactly at all 30 steps (runs L1-20261108-0144-cpu-epoch-{noresume,resume}).
- Clock stays on. Same 8,382 steps and 549.3M tokens, same warmup and cosine schedule.
That's three changes at once (size, LR, loader), so the result can't be credited to any one of them. I'm doing it deliberately: there's quota for one try, and the goal now is passing Level 1, not attributing the pass.
Smoke test (rule: short smoke test before any long run): the exact main-run configuration (DDP + compile + clock + epoch loader at 88M) on Kaggle. It stops itself after its 10-minute time limit, then runs the final full-validation eval, samples and judge at the new size.
Prediction:
- No crash, no out-of-memory error, no relaunch without compile.
- Peak memory 7.5-9.5 GB per GPU. The 28M model used 7.1 GB at twice the micro-batch.
- Speed 40k tokens/s (range 36-46k). The 88M model costs 3.1x the compute per token, and the 28M model ran at 120k. At 40k, the full run is about 3.8 h of training.
- Train loss falls through warmup with no spike. At step 200, the mean of the last 10 steps is at or below 2.33, the 28M main run's value at the same step.
Cost: about 0.3 GPU-h, $0.

## 2026-11-08 02:12 · L1 · result: smoke test L1-20261108-0149-big-smoke
(Result for the 01:49 entry.)
Actual (from the run's metrics.jsonl, events.jsonl and the kernel's session.json): the session started 10 s after the push, the first step was at 126 s (compile), and the 10-minute limit stopped the run cleanly at step 214 (14.0M tokens). The final full-validation eval (104 s), samples, judge and checkpoint all ran. Exit code 0. Session 769 s.
- Crash / out of memory / relaunch: none.
- Peak memory 8.38 GB per GPU.
- Speed: median 49.6k tokens/s (1.33 s per step).
- Train loss (mean of 10 steps), 88M vs 28M main run at the same step: step 50: 4.17 vs 4.37; step 100: 3.05 vs 3.16; step 150: 2.51 vs 2.61; step 200: 2.22 vs 2.33. No one-step rise above 0.15 after step 50. Max grad norm from step 100 on: 1.25 (28M run: 1.53).
- The epoch loader logged 1,072,896 of 1,072,967 windows (99.99%) for the full run.
Was I right?
- No crash, memory 7.5-9.5 GB, smooth warmup, at or below 2.33 at step 200: right on all four.
- Speed 40k (range 36-46k): wrong, in the good direction: 49.6k. A wider model uses the T4's matrix units better than I allowed for. The full run is about 3.1 h of training, not 3.8.
What I learned: the 88M configuration is safe to run long, and it's ahead of the 28M model from the first steps even at the lower LR.

## 2026-11-08 02:12 · L1 · L1-20261108-0212-big-clock (the Level 1 retry)
Decision: The configuration from the 01:49 entry, unchanged from the smoke test except for the time limit (the kernel code is byte-identical to the smoke test's). 88.35M params, 12 x 768, epoch loader, LR 1.0e-3, warmup 200, cosine to 10%, clock on, seed 1337, 8,382 steps x 65,536 tokens = 549.3M tokens, one pass over the training text. Eval every 500 steps, latest.pt every 15 min. The run stops itself at 240 min; the kernel is capped at 265 min.
Hypothesis: the failures in the reading test were about tracking characters and objects through a story. A network 3.2x larger, which also sees every story once, tracks them well enough that most stories hold together.
Prediction:
- Speed 49.5k tokens/s (range 47-52k). About 3.1 h of training, about 3.3 h of session.
- Final full-val loss 1.06 (range 1.02-1.10), about 0.38 bits per byte; the 28M model got 1.145. The 88M model is 0.10 ahead at step 200; I expect the gap to narrow but not close.
- End-of-story token loss about 0.39 (0.42 for the 28M). Clock head error about 0.29.
- Reading test (21:14 protocol, unchanged: same prompt, temperature 0.8, seeds 7-16): 6 of 10 meet all three. Range 4-9. Honest note: my most likely outcome is still just short of the 7 needed. Going from 1.34 to 1.145 bought about 2 stories; another 0.08 of loss would buy about 1 by that trend, and I'm betting the extra depth helps coherence more than the loss alone shows. This is the number I'm least sure of.
Contingency, if it fails: the final checkpoint carries the optimizer state, so a second epoch could continue from it. That costs about 3.3 GPU-h, which would come out of Level 3's budget, so it would be the owner's call.
Cost estimate: about 3.4 GPU-h, $0. Week total afterwards about 6.7 of 30.

## 2026-11-08 06:04 · L1 · result: training numbers for L1-20261108-0212-big-clock
(Result for the 02:12 entry. The reading test is the next entry.)
Actual (from the run's eval.jsonl, metrics.jsonl, events.jsonl and the kernel's session.json): the session started 8 s after the push and ran all 8,382 steps (549.3M tokens, one pass over the training text) in 13,236 s. It finished under its own 240-minute limit, then ran the final full-validation eval, samples and checkpoint. Exit code 0, no relaunch. Session 13,257 s.
- Final full-val loss 1.0648, 0.384 bits per byte. The 28M model got 1.1447 / 0.413.
- By position: end-of-story token 0.355 (28M: 0.423); tokens 2-16 before the end 1.062 (28M: 1.151); 257 or more before the end 0.990 (28M: 1.091).
- Clock head error 0.292 (28M: 0.299; the constant-guess baseline is 0.750).
- Speed: median 45.0k tokens/s, flat from the first 500 steps to the last (every 500-step median between 44.9k and 45.0k). Peak memory 8.38 GB per GPU.
- Stability: no one-step loss rise above 0.15 after step 60. Grad norm above 1.0 at 11 steps after step 100; the largest was 2.51 at step 1039, and the loss did not jump with it.
- The automatic story judge ended at 0.973, the same as the 28M model. It no longer separates the two.
- Cost: 3.704 GPU-h (session start to COMPLETE seen), $0. Week total 7.03 of 30.
Was I right?
- Final loss 1.06 (range 1.02-1.10): right, 1.065. Bits per byte about 0.38: right. "The gap narrows but doesn't close": right, 0.10 at step 200 and 0.080 at the end.
- Clock error about 0.29: right, 0.292.
- End-token loss about 0.39: wrong in the good direction, 0.355. The bigger model knows better when a story is over.
- Speed 49.5k (range 47-52k): wrong, 45.0k. It was flat for the whole run, so it wasn't heat or slowdown over time. The smoke test got 49.4k on the same code, so the likeliest cause is that Kaggle gave this session a slower pair of T4s. That made training 3.6 h instead of 3.1, and the cost 3.70 GPU-h instead of 3.4.
What I learned: the loss went where I said it would. The speed of a single smoke test is not a promise. For the Level 3 budget I'll plan with about 10% less speed than a smoke test shows.

## 2026-11-08 06:06 · L1 · result: reading test for L1-20261108-0212-big-clock: FAIL, 4 of 10
(Result for the reading-test prediction in the 02:12 entry. Protocol from 21:14, unchanged: final checkpoint, step 8382; prompt "Once upon a time"; temperature 0.8; seeds 7-16; at most 500 new tokens; generated on this PC with tools/tell_stories.py.)
Actual: the model ended all 10 stories itself (118 to 186 tokens). My grades are in claude_reads.jsonl, and every story and grade is in reading_sheet.md. Same rules as the calibration and the 28M run: a phrase-level slip with the meaning clear and the events intact is a borderline pass, and a slip that changes facts fails.
- Named character 8, plot holds together 6, ending 9.
- All three: 4 of 10 (seeds 8, 11, 14, 16). Two are clean (11, 16), two borderline (8: "he forgot about the ball" with no ball in the story; 14: "a small hole... but it is big", and the bunny's own lost way home is dropped).
- Fails, by reason. No name: seeds 10 and 12 both open "there was a [bunny / onion]" and never name it; the 28M run also had exactly 2 of these. Plot: seed 7 (the rain washes the umbrella while they sit in the house), seed 9 (the counter in hide and seek becomes the one who is found), seed 13 (a bottle, then a jar, then a cup taken out of the toilet), and seed 15 (cleaning a toy boat cures a bird that is outside the window, and it then flies around the room).
- Against the earlier readings: the 22:40 calibration (1.34 loss) got 2 of 10, the 28M model (1.145) got 4 of 10 with none clean, and this one (1.065) got 4 of 10 with 2 clean.
Was I right? No. I predicted 6 (range 4-9), and it landed at the bottom of my range. Level 1 needs 7, so by the protocol this is a fail. The owner has the final word.
What I learned:
- Another 0.08 of loss did not buy more passing stories. My trend from the 02:12 entry (about 1 story per 0.08) was wrong, or it is lost in the noise. With 10 stories, a model that truly passes half the time scores anywhere from about 2 to 8. So this test can't tell 4 from 6, but it does say the model is not reliably at 7.
- The failures are the same kind as the 28M model's: keeping track of where things are, who is doing what, and why one event causes the next. Scale moved the writing (fewer garbled lines, two clean stories), not the bookkeeping.
- The contingency (a second epoch, about 3.7 GPU-h at this run's speed) would lower the loss by a few hundredths. Given the above, I don't expect it to reach 7 of 10. I'd rather keep those hours for Level 3. It's the owner's call.

## 2026-11-08 13:30 · L1 · decision: Level 1 closed as a FAIL
The owner read the 88M model's 10 stories and gave the verdict: fail, move on (logged in interventions.jsonl). No second epoch. Level 1 ends here with two honest attempts: 28M parameters, 4 of 10; 88M parameters, 4 of 10. The best checkpoint is runs/L1-20261108-0212-big-clock/ckpt/step_8382.pt (val 1.065). My own idea, the narrative clock, passed its A/B test (NOTEBOOK 2026-11-07), and the bigger model's clock error was 0.292. Level 1 used 7.03 GPU-h of this week's 30, and $0.
What carries forward: the tokenizer code, the model, the epoch loader and the training loop. Plan Level 3 at about 10% below smoke-test speed. And the lesson that lower loss does not by itself buy coherent long-range bookkeeping at this scale.

## 2026-11-08 14:21 · L2 · plan and pass protocol (fixed before any Level 2 run)
Decision: Level 2 trains a model only on chess games written as text, with no rules given to it anywhere.
Data: Lichess's public database of rated standard games (database.lichess.org, CC0), months 2017-01 and 2016-06. My own parser reads the PGN text (headers and movetext) and keeps blitz-or-slower games between rated players, ended normally or on time, at least 10 plies. The exact rating filter gets fixed once I see the counts. python-chess is not used anywhere in the data pipeline: it is only the referee (legality checks, match play) and, for the probe, the source of the true board to compare the model's guesses against. The model never sees a board, a list of legal moves, or anything but the game text.
Text format: one game per record, in standard PGN movetext with clock comments removed: "1. e4 e5 2. Nf3 Nc6 ... 1-0". The end-of-text token separates games. Training windows (512 tokens) start at the beginning of a game, so the model always sees a game from move 1.
Tokenizer: my own BPE code from Level 1 (src/bpe.py), retrained on the chess text, with a pre-tokenizer that splits on spaces only. So BPE can never merge across two moves, and most moves become one token (" Nf3", " exd5", " O-O").
Model: the Level 1 network (pre-norm, RoPE, RMSNorm, SwiGLU, tied embeddings), 12 layers x 512, context 512, about 38M non-embedding parameters.
My own idea for this level, "spelled-out move embeddings": a whole-move token like " Nf3+" hides its spelling from the network, so " Nf3" and " Bf3" start out as unrelated as " Nf3" and " a6". The idea adds small learned tables for the parts of a move that can be read off its spelling: piece letter, destination square, capture mark, check/mate mark, promotion piece, disambiguation, castling side, move number. Each token's embedding is its own vector plus the sum of its parts. The embeddings are tied, so the output layer gets the same sharing. This is pure string reading of the token, the same information a character-level model sees; it says nothing about which moves are legal. Summing sub-part embeddings is related to subword embeddings in NLP (fastText); using move-spelling parts to give whole-move chess tokens shared structure is my design. The part tables start at zero, so both arms start from identical weights.
A/B test (rule 4): one Kaggle session, one model per T4, same seed, same initial weights, same data order. Arm A baseline, arm B with spelled-out embeddings. Both are full-size runs, and the session costs the same quota as one DDP run, because Kaggle charges per session hour.
Decision rule (fixed now): the arm with the higher legal-move rate on held-out positions at its final eval plays the match (tie within 0.1 points: lower val loss). The idea counts as working if arm B has (a) a lower illegal-move rate than arm A and (b) a val loss no worse than A's + 0.002. One seed pair only, because of the quota, so a small difference is weak evidence.
Pass protocol (fixed now):
1. Match: 20 games against Stockfish 19, Skill Level 0, 0.1 s per move, on this PC. The model plays white in games 1-10 and black in 11-20. It plays greedily (most likely next move given the game so far, in the training text format). The referee (python-chess) checks every move. If the model's move is illegal, that counts as an illegal move, and it gets up to 4 more tries sampled at temperature 1.0. If none is legal, it forfeits the game. Games end by python-chess's rules (mate, stalemate, insufficient material, threefold repetition or 50-move rule counted as a draw); a game reaching 300 plies is adjudicated a draw.
2. Legal moves: "legal" means python-chess's parse_san accepts the model's move text as one legal move in that position. I also report the stricter "exact" rate (text identical to the official SAN, including x, + and # marks). Pass needs at least 99% of the model's first attempts legal over its first 1,000 moves in match games. If the 20 games hold fewer than 1,000 model moves, more games are played (same settings) until 1,000 is reached; those extra games don't count toward the win.
3. Pass = at least 99% legal AND at least one win in the 20 games.
Board probe (fixed now): linear probes trained on the network's hidden state at the last token of each move, at every layer (embedding output and after each of the 12 blocks), predicting what stands on each of the 64 squares (13 classes). Trained on positions from held-out games, scored on other held-out games. Baselines: the same probe on a randomly initialised copy of the network, and a guess that knows only the ply number (the most common contents of each square at that ply). The board-in-its-head claim needs the trained network's best layer far above both.
Prediction:
- Held-out legal-move rate at the end: baseline 98.8%, spelled-out 99.1% (honest range for both: 97.5-99.7%).
- Val loss: spelled-out about 0.01 nats per token lower.
- Match legality: about 98.5%, lower than on human positions, because Stockfish's blunders lead to positions unlike the training games. This is the criterion I think is most likely to fail.
- Wins: Stockfish level 0 at 0.1 s blunders often. I predict 6 wins in 20 (range 2-12).
- Probe: best layer around two thirds of the way up, near 95% per-square accuracy, against about 75% for the ply-only guess.
Cost estimate: $0. Smoke test about 0.3 GPU-h, A/B session about 2.6 GPU-h. Level 2 total about 3 GPU-h, week total about 10 of 30 afterwards, leaving about 20 for Level 3 (plus the 20 TPU hours).

## 2026-11-08 14:40 · L2 · data and tokenizer (result), CPU smoke tests
Data (from data/lichess/parse_*.json and data/chess/tok/meta.json): 2017-01 had 10.68M games, 7.71M passed the header filters (2.77M bullet dropped); 2016-06 had 6.14M, 4.28M passed. I fixed the rating filter at both players >= 1700: 4.37M games. (>= 1800 would have left 2.79M, which the A/B arms would have seen about 1.4 times each; 1700 gives fresh games all run, and lower-rated games have more of the odd positions Stockfish's blunders will create.) Split, seed 2026: 4,343,601 train games (491.8M tokens, 74.0 plies and 113 tokens per game), 20,000 validation, 10,000 test. 567 training games that were move-for-move copies of a held-out game were dropped.
Tokenizer: 6,000 merges, vocab 6,257. The chess text has only 13,244 distinct words in 487M (moves, move numbers, results). 99.971% of move occurrences are a single token. Merge 1 and the full list are in tokenizer/chess/merges.txt.
CPU smoke tests (all passed): L2-20261108-1434-cpu-smoke (tiny model with spelled-out embeddings, 40 steps): first loss 8.745 vs ln(6257) = 8.741 for a uniform guess, every log file written. Crash-resume: L2-20261108-1436-cpu-resume crashed on purpose at step 25 and resumed from latest.pt; its losses match the uninterrupted L2-20261108-1436-cpu-noresume at all 40 steps (max difference 0.0). The evaluator scores a fake model that replays the human moves at 100% legal; an untrained network scores 3-5%. The match tool and probe tool ran on the smoke checkpoint.

## 2026-11-08 14:40 · L2 · Kaggle smoke test L2-<push time>-smoke-{base,spelled}
Decision: the exact main-run setup on Kaggle: two single-GPU runs side by side (GPU 0 baseline, GPU 1 spelled-out embeddings), 12 x 512, context 512, batch 64 x 512 = 32,768 tokens, torch.compile, LR 1.5e-3, warmup 300, the main run's 380M-token schedule. Each run stops itself after 8 minutes, then runs the final eval as the main run will (full validation set, legality on 2,000 test games).
Why: rule 7, and to measure speed, memory and eval time, which size the main run.
Prediction:
- No crash, no out-of-memory error, no relaunch without compile, on either arm.
- Peak memory 9-11 GB per GPU (the 8-layer Level 1 model used 7.1 GB at this batch).
- Speed 46k tokens/s per GPU (range 40-52k); the spelled-out arm within 3% of the baseline.
- At the 8-minute stop (about 22M tokens), legal-move rate on held-out positions about 60% (range 30-90%), and the spelled-out arm's train loss about 0.03 lower than the baseline's.
- Final eval (2.26M validation tokens plus about 148k legality positions) about 2 minutes.
Cost estimate: about 0.35 GPU-h, $0.

## 2026-11-08 15:09 · L2 · result: Kaggle smoke test L2-20261108-1443-smoke-{base,spelled}
(Result for the entry headed "14:40 · Kaggle smoke test". Correction: both "14:40" entries were appended at 14:38 by the clock; I typed the header time ahead of the clock again. The push itself was at 14:43, which is in the run IDs.)
Actual (from each run's metrics.jsonl, eval.jsonl, events.jsonl and the kernel's session.json): the session started 44 s after the push; both runs hit their 8-minute limit, ran the final eval and exited 0. Session 728 s.
- Crash / out of memory / relaunch: none. Both arms started from the same loss at step 1 (8.8389), so the identical-start design works.
- Peak memory 10.09 GB per GPU, both arms.
- Speed: baseline (GPU 0) median 44.5k tokens/s, 239 steps (7.8M tokens); spelled-out (GPU 1) 38.7k, 201 steps (6.6M tokens), 13% slower.
- Train loss at equal steps, spelled-out vs baseline (mean of 10 steps): step 50: 5.06 vs 5.40; step 100: 3.47 vs 3.55; step 200: 2.75 vs 2.84.
- Legal-move rate on 2,000 test games (147,241 positions) at the stop: baseline 65.8% at step 239, spelled-out 61.2% at step 201 (different step counts, so not comparable). By ply: about 95% in the first 20 plies, 60% at plies 21-60, 46% after.
- Final eval took 147 s (baseline) and 166 s (spelled-out); an in-run eval (300 games) takes about 32 s.
- Gradient norms: the baseline had one spike of 9.4 at step 150 (loss 3.24 -> 3.47, back the next step). The spelled-out arm's norm sat at 2-16 through warmup and above 2 most of the time after. Its first-step norm was 24.1 against 12.4.
Was I right?
- No crash, memory 9-11 GB: right on both.
- Speed 46k (range 40-52k): baseline right (44.5k). "Spelled-out within 3%": wrong, 13% slower.
- Spelled-out loss 0.03 lower at equal steps: right direction, larger: 0.09 lower at step 200, 0.34 at step 50.
- Legality about 60% at the stop: right (66% and 61%).
- Final eval about 2 minutes: close (2.5-2.8 min).
What I learned, and what changes before the main run (all from this evidence, before any A/B result exists):
1. The parts table's rows are shared by thousands of tokens, so they collect large gradients. With one global clipping norm, those rows set the clip for the whole network, so the two arms' shared weights were clipped differently. That's a confound in the A/B. Fix: the shared network is clipped exactly as in the baseline, and the parts table is clipped on its own. grad_norm in metrics.jsonl is now the shared network's norm in both arms; the parts table's is grad_norm_parts.
2. Speed: summing each token's part vectors by gathering is slow to differentiate. The same sum as one matrix product (a fixed count matrix times the parts table, in fp32) gives identical embeddings (difference 0.0 on the smoke checkpoint) and is 5x faster on CPU. Level 1's clock arm was also on GPU 1 and also slower (5-8%), so part of the gap may be the GPU slot. The main run puts the spelled-out arm on GPU 0 to tell the two apart.
3. The baseline's code path is unchanged from this smoke test. The spelled-out arm's changes were checked on CPU (L2-20261108-1508-cpu-smoke2) but not on a T4. The kernel's retry-without-compile covers a failure in the first 20 steps.
4. The smoke test never reached the peak LR (warmup 300, stopped at 239), and the baseline spiked once at 0.75x peak. For the main run I'm lowering the peak LR to 1.2e-3 and lengthening warmup to 500, the same for both arms. A 12-layer model at 1.5e-3 hasn't been tested anywhere in this project, and a divergence late in the run would cost both arms.
Cost: 0.233 GPU-h (ledger). Week total 7.26 of 30.

## 2026-11-08 15:09 · L2 · main A/B run L2-<push time>-{spelled,base}
Decision: one Kaggle session, both arms in parallel: spelled-out on GPU 0, baseline on GPU 1. Same seed (1337), same initial weights, same data order. 12 x 512, context 512, batch 32,768 tokens, LR 1.2e-3, warmup 500, cosine to 10%, weight decay 0.1, clip 1.0, torch.compile. 350M tokens each = 10,681 steps, 0.71 of an epoch (no game is seen twice). Sized from the smoke speed minus 10% (the Level 1 lesson): at 40k tokens/s that is 146 min of training. Each run stops itself at 175 min of wall time; the kernel is capped at 200 min. Eval every 1,000 steps (val loss on 1M tokens, legality on 300 test games), final eval on the full validation set and 2,000 test games. latest.pt every 15 minutes, 12 log-spaced checkpoints.
Decision rule: unchanged from the 14:21 entry.
Prediction:
- Both finish all 10,681 steps, no crash. Speed about 42k tokens/s for each arm. If the GPU slot was the cause, the baseline on GPU 1 is now the slower one.
- Final full-val loss: baseline 1.20 (range 1.10-1.35); spelled-out 0.01 lower. Most of the loss is the move tokens; move numbers are nearly free.
- Final legal-move rate on 2,000 test games: baseline 98.8%, spelled-out 99.1% (from the 14:21 plan, unchanged). Human-move match about 45% for both.
- The spelled-out arm's early lead (0.09 at step 200) shrinks to about 0.01 by the end, because sharing helps most while the token vectors are still untrained.
Cost estimate: session about 160 min = 2.7 GPU-h, $0. Week total afterwards about 10.0 of 30.

## 2026-11-08 19:16 · L2 · result: main A/B run L2-20261108-1509-{spelled,base}
--- result (added after) ---
Run IDs: L2-20261108-1509-spelled (GPU 0) and L2-20261108-1509-base (GPU 1), kernel kaggle-user/ep04-l2-main, pushed 15:09:36. Both arms ran all 10,681 steps (350.0M tokens each) with no crash, NaN or restart. Session 8,671.5 s by the kernel's clock (15:09:46 to 17:34:23), logged as 2.43 GPU-h (ledger.jsonl), $0. Week total 9.69 of 30 GPU-h. My background watcher stopped at 17:19 because this PC went to sleep. I only saw COMPLETE at 19:00, so this time the quota figure comes from the kernel log span plus one minute, not from the watcher.
Actual (final eval: full validation set, 2.59M tokens; legality on 2,000 held-out test games, 147,241 positions, the same positions for both arms):
| | baseline | spelled-out |
|---|---|---|
| val loss (nats/token) | 1.1228 | 1.1105 |
| legal-move rate | 98.504% (about 2,203 illegal) | 98.597% (about 2,066 illegal) |
| exact-SAN rate | 97.96% | 98.05% |
| human-move match | 48.78% | 49.12% |
| legal, plies 1-20 / 21-60 / 61+ | 99.96 / 98.59 / 96.87% | 99.97 / 98.67 / 97.06% |
| median speed | 45.9k tok/s (GPU 1) | 50.3k tok/s (GPU 0) |
| peak memory | 10.09 GB | 10.10 GB |
Val loss gap (baseline minus spelled-out) at each eval: 0.040 at step 1k, 0.024 at 2k, 0.016 at 3k, 0.016 at 4k, 0.014 at 5k, 0.013 at 6k-8k, 0.013 at 9k, 0.012 at 10k, 0.012 final. Spelled-out was ahead at all 11 evals after step 0. The legal-rate gap on the 300-game evals was +0.06 to +0.20 points, with spelled-out ahead at 10 of the 11.
Decision rule (14:21 entry), applied as written:
- Match model: the legal-rate gap is 0.093 points, inside the 0.1-point tie band, so the lower val loss decides: spelled-out (1.1105 vs 1.1228). It would have been spelled-out without the tie rule too. The match uses runs/L2-20261108-1509-spelled/ckpt/step_10681.pt.
- Does the idea work? (a) Spelled-out has the lower illegal-move rate: 1.403% vs 1.496%, about 6% fewer illegal moves. Yes. (b) Its val loss is 0.012 lower, so it is not more than 0.002 worse. Yes. By the pre-registered rule, the idea works.
- How much to trust that: one seed pair. The val-loss gap is small (1.1%) but the same sign at every eval. The legality gap is about 137 moves out of 147k on identical positions: real for these two runs, but I can't rule out a seed-to-seed spread of that size. Fair claim for the video: "a small, consistent gain, one seed pair".
Was I right?
- No crash, all steps: right.
- Speed about 42k for each: wrong, both were faster (50.3k and 45.9k). The GPU-slot guess was right. In the smoke test the arm on GPU 1 (spelled-out) was the slow one; now the arm on GPU 1 (baseline) is the slow one. So GPU 1 of these Kaggle machines runs about 9-13% slower, and the speed numbers say nothing about the idea's cost. Loss at equal steps is not affected: same data order, same step count.
- Baseline val 1.20 (range 1.10-1.35): actual 1.1228, inside the range, better than my point guess. Spelled-out 0.01 lower: actual 0.012 lower. Right.
- Legal rate 98.8% / 99.1%: actual 98.50% / 98.60%. Both too high by 0.3-0.5 points, and the spelled-out model is below the 99% bar on held-out human games. The misses are concentrated late in the game (ply 61+: 97.1%). Ply 1-20 is 99.97%.
- Human-move match about 45%: actual 49%. Better than I guessed.
- Early lead shrinks to about 0.01: right (0.040 at step 1k down to 0.012 at the end).
- Cost 2.7 GPU-h, week about 10.0: actual 2.43 and 9.69.
- Small note: my 15:09 entry said the run is 0.71 of an epoch. The loader's own count is 683,584 of 1,103,086 windows = 0.62 of an epoch: game-aligned windows waste the tail of each window that can't fit the next game. Either way no game is seen twice.
What I learned: the spelled-out embeddings give a small but steady gain at no measurable cost. The bigger news is that 350M tokens isn't enough for 99% legality on human games: the model is at 98.6% and still climbing (98.58% at step 9k, 98.70% at 10k, 98.76% at the end on the 300-game set). The match is the real test, and it is now more likely to fail on legality than on wins.

## 2026-11-08 19:13 · L2 · match vs Stockfish level 0, checkpoint L2-20261108-1509-spelled step 10681
Corrections to the entry just above (left unedited):
- Its header says 19:16. The system clock read 19:12 when I wrote it.
- The sentence "The legal-rate gap on the 300-game evals was +0.06 to +0.20 points, with spelled-out ahead at 10 of the 11" is wrong; I wrote it from memory. From eval.jsonl, spelled-out minus baseline at steps 1k-10k and 10,681: +1.12, +0.64, +0.33, +0.10, +0.13, +0.26, +0.30, +0.10, +0.16, +0.17, +0.21 points. Spelled-out was ahead at all 11; the gap is large early and about +0.1 to +0.3 from step 4k on.
Decision: play the pre-registered match (14:21 entry, protocol unchanged) with runs/L2-20261108-1509-spelled/ckpt/step_10681.pt: Stockfish 19, Skill Level 0, 0.1 s per move, 20 games, the model white in 1-10 and black in 11-20. Greedy play; up to 4 more tries at T=1.0 after an illegal move, then forfeit; 300-ply cap. Extra games until 1,000 model moves (they don't count toward wins). Runs on this PC's CPU (tools/play_stockfish.py, 6 workers, seed 7), output in runs/L2-20261108-1509-spelled/games/final_sf0/. No GPU.
Why this checkpoint and only this one: it is the one the decision rule picked. Skill Level 0 picks moves at random among weaker ones, so two matches with the same model give different results. Playing the same checkpoint again until it passes would be fishing. So this first match is the one that counts for this checkpoint.
Prediction:
- First-attempt legality over the first 1,000 model moves: 98.0% (range 96.5-99.3). That is lower than the 98.6% on human games, because games against a weak engine go places human games don't. I put about a 20% chance on reaching 99%, so I expect Level 2 to fail on legality with this checkpoint.
- Exact-SAN rate about 0.5 points below the legal rate.
- Results in the 20 games: 7 wins (range 2-13), about 3 draws, the rest losses, with at most 1 forfeit. The model matches the human move 49% of the time in 1700+ games. Stockfish Level 0 at 0.1 s throws away material often, but it still punishes hanging pieces.
- Most illegal attempts will be "no_such_move" (the piece it names can't reach that square: the board in its head is wrong) rather than "king_in_check" or "ambiguous". They will cluster after ply 60.
Contingency, fixed now, before the match:
- If legality is below 99% or there is no win, Level 2 is a FAIL for this checkpoint, and that is reported as is.
- I then get one more try: a continuation run of the spelled-out model from step 10,681, weights and optimizer state, with no change to the model. It trains on more game text: the 38% of training windows it hasn't seen yet, plus fresh Lichess months if I can prepare them on the CPU in time. Warm restart at a lower LR, budget at most 3 GPU-h. It gets a notebook entry with its own prediction before it starts.
- Then one new 20-game match, same protocol, all games count. Both matches get reported. No third try in Level 2, because the rest of the quota (about 20 GPU-h) is for Level 3.
Cost estimate: $0, 0 GPU-h. About 10-20 minutes of CPU.

## 2026-11-08 19:18 · L2 · result: match vs Stockfish level 0, L2-20261108-1509-spelled step 10681: FAIL on legality
--- result (added after) ---
Files: runs/L2-20261108-1509-spelled/games/final_sf0/ (game_01..32.pgn with IllegalMoves headers, moves.jsonl, games.jsonl, summary.json, illegal_kinds.json); console in runs/L2-20261108-1509-spelled/games_final_sf0_stdout.log. 32 games in 80 s on this PC's CPU, 0 GPU-h, $0.
Actual:
- The 20 match games: 5 wins, 5 draws, 10 losses. 3 of the losses are forfeits (no legal move in 5 tries). Wins: games 5 and 10 as white, 13, 19 and 20 as black. Game 20 is a mate in 18 moves (14...Bxh3 15.gxh3 Qxh3 ... 18...Qh1#). Draws: 4 by threefold repetition, 1 stalemate.
- The 20 games held 756 model moves, so 12 extra games were played to reach 1,000 (they don't count toward wins).
- First-attempt legality over the first 1,000 model moves: 979 of 1,000 = 97.9%. Exact SAN 96.1%. All 18 moves that were legal but not exact differ only in a missing or extra + or # mark.
- Pass = at least 99% legal AND at least one win: the win part passes, the legality part fails. Level 2 is a FAIL for this checkpoint.
- The 21 illegal first attempts in the first 1,000 by kind (tools/illegal_kinds.py):
  - no_such_move 13 (the named piece can't reach that square)
  - king_in_check 3
  - ambiguous 3
  - not_a_move 2
- By game stage: 0 illegal in plies 1-20 (284 moves), 10 in plies 21-60 (499 moves, 2.0%), 11 at ply 61+ (217 moves, 5.1%).
- The forfeits: in two of the three, the model tried to write the result instead of a move. Game 16: "1-0" five times, in a position where it is a rook down. Game 14: "1-0" among its tries. It learned that games end with a result string, and in a lost position it resigns, which the protocol counts as an illegal move. A typical no_such_move: in extra game 30 it tried "Qg2#" four times, with its queen on c2 and its own pawn on f2 in the way. The board in its head has the queen's line open.
Was I right?
- Legality 98.0% (range 96.5-99.3): actual 97.9%. Right, and the expected FAIL came true.
- Exact about 0.5 points below legal: actual 1.8 points below. Wrong. The check marks are sloppier than I thought.
- 7 wins (range 2-13): actual 5, inside the range. About 3 draws: actual 5. At most 1 forfeit: actual 3, wrong. I didn't think of the model resigning.
- Most illegal attempts are no_such_move and they cluster late: right (13 of 21; 11 of 21 come after ply 60, from 22% of the moves).
What I learned: the model plays real chess (it beat Stockfish level 0 five times), and its opening knowledge is clean: no illegal move in the first 20 plies of 32 games. What it lacks is board tracking deep into a game. Illegal moves are 2.0% in plies 21-60 and 5.1% after ply 60. That matches the held-out human games (97.1% legal after ply 60). It's the same weakness, made worse by long games against a weak engine. More training data is the direct fix: on the 300-game eval, legality was still rising at the end of the run.
Next: the contingency fixed in the entry above. One continuation run of this model on more game text, then one new match.

## 2026-11-08 19:19 · L2 · board probe and skill probe on L2-20261108-1509-spelled step 10681
Correction to the entry just above (left unedited): in game 16 the model was a queen down, not a rook down. The position (4r1k1/pp3pp1/3N1n1p/2P5/1P6/P6Q/5P1P/3R1K2 b) has white with queen, rook and knight against black's rook and knight, five pawns each.
Decision: run the pre-registered board probe (14:21 entry) on the checkpoint that played the match, with that run's step-0 checkpoint as the untrained baseline. tools/probe_board.py: a linear probe per layer (0 = embedding output, 1-12 = after each block) on the hidden state at the last token of each move, 64 squares x 13 classes, two labelings ("absolute": white/black pieces; "relative": mine/theirs from the side that just moved). 1,200 held-out test games to train the probes, 300 others to score them (test_games.jsonl from index 3000, so no overlap with the 2,000 legality games). 30 epochs, LR 1e-2, the settings fixed after the smoke checkpoint showed the earlier 8-epoch probe was under-trained. Output: runs/L2-20261108-1509-spelled/probe/{probe_layers.json, probe_demo.json, probes.pt}.
Also the optional skill probe (camera spec, Level 2): tools/probe_skill.py, ridge regression from each layer's hidden state averaged over plies 11-40 to the mean rating of the two players (from the PGN headers, which the model never saw). 3,000 games to fit, 1,000 others to score (test_games.jsonl from index 5000, games of at least 40 plies). Baselines: guessing the training mean, and the untrained network.
Prediction:
- Board probe: best layer about two thirds of the way up (layer 7-9 of 12), per-square accuracy 94% (range 88-97%) on the absolute labeling. Ply-only guess about 75% (74.8% on the smoke check). Untrained network's best layer about 78%: the embedding alone already says which piece just moved where. Relative labeling 1-2 points higher than absolute at the best layer. Whole-board-exactly-right under 15%. The embedding layer (0) is the trained network's worst.
- On the demo game, the probe gets the last-moved piece's new square nearly always right. It gets squares wrong in the parts of the board where nothing has moved for a while.
- Skill probe: weak. Best trained layer correlation with the true mean rating about 0.35, test MAE about 115 points against about 128 for guessing the mean (rating SD about 164). The untrained network gets about 0.25, because opening choice alone carries some rating information. If the trained network beats the untrained one by less than 0.05 in correlation, I won't claim the model "knows" player strength.
Cost estimate: $0, 0 GPU-h, CPU only: about 15-30 minutes each.

## 2026-11-08 19:45 · L2 · contingency run: continue L2-20261108-1509-spelled on more games, L2-<push time>-cont
Decision: the one extra try fixed in the 19:13 entry. The spelled-out model continues from step 10,681 (weights, AdamW state and loss-scaler state from ckpt/step_10681.pt; architecture unchanged) on a bigger set of game text. The step count and LR schedule start fresh. Its final checkpoint, and only that one, plays one new 20-game match under the unchanged 14:21 protocol.
Data (tools/extend_chess.py, data/chess/tok2, Kaggle dataset kaggle-user/ep04-chess-tok2):
- Two new Lichess months, 2017-02 and 2017-03: 21.5M games downloaded, 15.5M pass the header filters, 4,970,315 with both players >= 1700. 681 identical to a held-out game were dropped.
- That is 568.2M tokens of games the model has never seen. They are followed by the first set's train.bin unchanged (491.8M tokens, 62% of its windows seen once).
- Total 1,060.1M tokens = 2,378,930 game-aligned windows.
- Same tokenizer, same validation set, same test games, so every number stays comparable with the first run.
- The run draws 650M tokens, 0.27 of an epoch of the union: about 350M from the new months and about 300M from the old ones, of which roughly 115M are repeats. No game is drawn twice within the run.
Config:
- DDP over both T4s with torch.compile. 64 x 512 tokens per GPU = 65,536-token batch (twice the A/B batch). 9,918 steps = 650.0M tokens.
- LR: warm restart to 5e-4 over 300 steps, cosine to 5e-5. The A/B peak was 1.2e-3 and it ended at 1.2e-4. A smaller peak keeps most of what it learned.
- Weight decay 0.1, clip 1.0, data-order seed 2027.
- Eval every 1,000 steps (val loss on the same first 1M val tokens, legality on the same 300 test games). Final eval on the full val set and the same 2,000 test games.
- The run stops itself at 150 min of training. Kernel cap 180 min.
Smoke test (rule 7): phase 0 of the same session is a 60-step run with the exact same setup (DDP + compile + init_from + the new data), as L2-<push time>-cont-smoke. With stop_on_fail, the main phase is skipped if it fails. New in this run: DDP with the spelled-out embeddings, and --init_from.
- CPU check L2-20261108-1920-cpu-init: the init event, weights and optimizer loaded, step-1 loss 1.04 instead of 8.8. I stopped it by hand during its slow CPU final eval; the eval code is unchanged from the A/B.
- src/data.py pack_starts now uses bisect on a list. It gives an identical window list on the old train and val sets (1,103,086 and 5,056 windows).
Prediction:
- Step-0 eval of the continuation is the old model again, on the same 1M val tokens and 300 games, so it should repeat the A/B's last 300-game eval: val 1.1134, legal 98.76% (within 0.001 for fp16 noise).
- The warm restart costs a little first: train loss up by about 0.02 around step 300, then below the old level by step 2,000.
- Speed about 88k tokens/s (1.9x the slower T4's 45.9k), so about 125 min of training. No crash.
- Final full-val loss 1.075 (range 1.055-1.095), against 1.1105 now.
- Final legal-move rate on the 2,000 test games: 98.95% (range 98.7-99.3), against 98.60% now. Human-move match about 50.3%.
- New match: legality 98.6% (range 97.8-99.4) over the first 1,000 model moves. I give about a 30% chance of reaching 99%, so my best guess is that Level 2 still fails. Wins 6 (range 2-12).
Cost estimate: $0. Session about 140 min = 2.35 GPU-h (at most 3.0 with the 150-minute stop). Week total about 12.0 of 30, leaving about 18 GPU-h plus the 20 TPU-hours for Level 3.

## 2026-11-08 19:45 · L2 · correction to the 19:45 contingency entry (before the push)
- "roughly 115M are repeats" is wrong arithmetic. 62% of the old windows were seen once, so about 0.62 x 300M = 185M of the 650M tokens are repeats. About 465M are new to the model.
- "No game is drawn twice within the run" should say no *window* is drawn twice. The packing can put the cut-off tail of one game in one window and the whole game at the start of the next.
The rest of that entry, and its predictions, stand unchanged.

## 2026-11-08 19:52 · L2 · result: board probe on L2-20261108-1509-spelled step 10681
--- result (added after) ---
Files: runs/L2-20261108-1509-spelled/probe/{probe_layers.json, probe_demo.json, probes.pt}, console in runs/L2-20261108-1509-spelled/probe_stdout.log. 88,359 positions to fit the probes and 21,571 to score them (1,200 and 300 held-out games). 1,880 s on this PC's CPU, 0 GPU-h.
Actual (test positions, absolute labeling unless noted):
| | per square | occupied squares | empty squares | whole board exactly right |
|---|---|---|---|---|
| ply-only guess | 73.7% | 40.5% | 92.4% | 0.8% |
| untrained network, best layer (2) | 79.9% | 57.2% | 92.7% | 7.3% |
| trained network, layer 0 (embedding) | 71.9% | 38.1% | 90.9% | 0.8% |
| trained network, layer 6 | 86.4% | | | |
| trained network, best layer (12, the last) | 90.5% | 81.2% | 95.8% | 18.0% |
| same, relative labeling (mine/theirs) | 91.0% | 81.9% | 96.1% | 21.4% |
- The trained network's accuracy rises with almost every layer: 71.9, 76.6, 79.9, 83.3, 83.6, 85.0, 86.4, 87.2, 87.9, 88.4, 88.4, 88.8, 90.5%. The untrained network peaks at layer 2 (79.9%) and drifts down after that.
- Probe train accuracy at layer 12 is 92.4% against 90.5% on test, so the probe isn't memorising. A linear readout just doesn't get more out of the state.
- Demo game (test game 4,200, 77 plies, 1868 vs 1972): all 64 squares right for the first 28 plies or so; 56-62 of 64 after that. The destination square of the move just played is right in 67 of 77 frames. Occupied squares untouched for 20+ plies are right 80.6% of the time; recently touched ones 81.0%.
Was I right?
- Best layer two thirds up (7-9): wrong. It is the last layer, and accuracy is still rising there.
- 94% per square (range 88-97): wrong in the point, inside the range at 90.5%.
- Ply-only about 75%: right (73.7%). Untrained about 78%: right (79.9%).
- Relative 1-2 points above absolute: half right, +0.4 points per square, +3.4 points on whole boards.
- Whole board under 15%: wrong, 18.0% (21.4% relative).
- Embedding layer worst: right (71.9%, below even the ply-only guess, because a token knows its own move but not the position).
- Demo: "last-moved piece's square nearly always right": wrong, 67 of 77. "Wrong squares where nothing has moved for a while": wrong, stale and fresh squares are equally good.
What I learned: there is a board in its head, but a blurry one. On occupied squares (the hard part) the trained network is at 81%, against 57% untrained and 41% for ply-only. It builds that picture layer by layer and is most complete at the very top, right where it picks the next move. The blur matches the match results: a model that knows four in five pieces' squares exactly plays legal moves 98% of the time but will sometimes move a queen through its own pawn.

## 2026-11-08 19:52 · L2 · correction to the board-probe result entry just above
- The demo game is test game index 4,201, not 4,200.
- "All 64 squares right for the first 28 plies or so; 56-62 of 64 after that" is wrong; I read it off every fourth frame. The first miss is at ply 2.
- From ply 29 on, the count alternates. After white's moves the probe gets 59-64 of 64; after black's moves, 45-57. Every fourth frame happened to land on white's moves.
- A guess to test, not a result yet: after black's move the next token is a move number (" 16."), which needs no board, so the network may not bother to build the board there. After white's move the next token is black's move, which needs one. I'll check it on all 21,571 test positions with the saved probes before claiming it.

## 2026-11-08 19:53 · L2 · exploratory: does the board appear only where the next token is a move?
Decision: a follow-up to the alternation found in the demo game. It is exploratory, not part of the pre-registered probe, and will be labelled that way in the video. tools/probe_parity.py, same checkpoint, same games (probe-fit 3000-4199, scoring 4200-4499 of test_games.jsonl), layers 6, 9 and 12, absolute labeling, same probe recipe (linear, 30 epochs, LR 1e-2).
Three reading points, one probe trained per reading point:
- (W) after white's move, at the last token of the move. The next token is black's move.
- (B) after black's move, at the last token of the move. The next token is a move number.
- (B+) the same board as (B), read one step later, at the last token of the next move number. The next token is white's move.
I also score the protocol's single joint probe separately on white and black positions, to confirm the alternation on all positions and not just the demo game.
Hypothesis: the network builds the full board where it has to choose a move, and lets it lapse where the next token is a trivial move number.
Prediction (layer 12, per-square / occupied-square accuracy):
- Joint probe: (W) about 93% / 88%, (B) about 88% / 75%.
- Separate probes: (W) 93% / 88%; (B) 88% / 76%; (B+) 93% / 87%, back up to (W)'s level.
- At layer 6 the gap between (W) and (B) is smaller (under 2 points), because the lower layers haven't specialised yet.
If (B+) is not at least 3 points (occupied squares) above (B), the hypothesis is wrong.
Cost estimate: $0, CPU only, about 10 minutes.

## 2026-11-08 19:56 · L2 · result: skill probe on L2-20261108-1509-spelled step 10681
--- result (added after) ---
File: runs/L2-20261108-1509-spelled/probe/probe_skill.json; console in runs/L2-20261108-1509-spelled/probe_skill_stdout.log. 3,000 games to fit, 1,000 to score (test_games.jsonl from 5000, at least 40 plies). Test-set rating SD 153.6. Ridge penalty chosen per layer on a split of the fitting games only.
Actual (test games, mean rating of the two players):
- Guessing the training mean: MAE 121.7 points.
- Trained network: correlation 0.20 at the embedding, rising to 0.34-0.35 at layers 6, 8, 11 and 12 (best 0.350 at layer 11). Best MAE 113.8 (layer 8).
- Untrained network: correlation 0.14 at the embedding, 0.27-0.29 at layers 3-12 (best 0.286 at layer 7). Best MAE 116.0.
- Trained minus untrained, best layer each: +0.064 in correlation, 2.2 points of MAE.
Was I right? Trained best about 0.35: right (0.350). MAE about 115: right (113.8). Mean guess about 128: close (121.7; this sample's SD is 154, not 164). Untrained about 0.25: close (0.29).
Verdict, by the rule I set before: the trained network beats the untrained one by 0.064, just over the 0.05 bar. So I can claim a weak signal and no more. With 1,000 test games the correlation itself is uncertain by about ±0.03, and picking the best of 13 layers flatters both networks. Fair line for the video: "it picks up a little about how strong the players are, mostly from which openings they choose; it is not a rating detector". Not a hero shot.

## 2026-11-08 20:00 · L2 · result (exploratory): the board is built where the next token is a move
--- result (added after) ---
File: runs/L2-20261108-1509-spelled/probe/probe_parity.json; console in runs/L2-20261108-1509-spelled/probe_parity_stdout.log. Same checkpoint, games, labels and probe recipe as the board probe. 0 GPU-h.
Actual, separate probe per reading point (per square / occupied squares / whole board exactly right):
| layer | W: after white's move (next: black's move) | B: after black's move (next: a move number) | B+: same board, at the move number (next: white's move) |
|---|---|---|---|
| 6 | 92.4 / 85.9 / 17.5% | 82.4 / 64.7 / 3.5% | 92.4 / 86.3 / 16.3% |
| 9 | 96.4 / 93.9 / 28.0% | 82.8 / 66.3 / 3.9% | 95.5 / 92.5 / 22.1% |
| 12 | 98.5 / 97.6 / 49.6% | 85.2 / 71.0 / 3.1% | 98.5 / 97.8 / 51.6% |
- The protocol's joint probe at layer 12 shows the same split: 97.1 / 94.7% on W positions, 83.9 / 67.4% on B positions. Its 90.5% headline is the average of the two.
- (B+ through the joint probe is meaningless, about 59%: that probe was never trained on move-number tokens.)
Was I right?
- Joint probe W about 93 / 88 and B about 88 / 75: the direction is right, the size wrong. W is higher (97.1 / 94.7), B is lower (83.9 / 67.4).
- Separate probes W 93 / 88: too low, actually 98.5 / 97.6. B 88 / 76: too high, actually 85.2 / 71.0. B+ 93 / 87, back to W's level: back to W's level, yes, but that level is 98.5 / 97.8.
- At layer 6 the W-B gap is under 2 points: wrong. It is already 10 points per square and 21 on occupied squares.
- The test I set (B+ at least 3 points above B on occupied squares) passes by 27 points. The hypothesis survives.
What I learned: the network does keep a board in its head, and a sharp one, but only where it needs it. Read at the moment it has to write a move, a linear probe recovers 98.5% of squares, 97.7% of occupied squares, and the entire board exactly right half the time. One token earlier, where the next token is a move number, the same board is a blur (71% of occupied squares): the network doesn't hold it there, and rebuilds it at the move-number token. This is exploratory: I found the split in the demo game and then tested it. But the prediction and the falsification test were written before the run. The video should say both: "the pre-registered probe says 90.5%; looking closer, it's 98.5% at the moment it moves".
Note on the skill-probe entry above: "mostly from which openings they choose" is my reading of the untrained baseline (0.29 from move-token statistics alone), not something I measured.

## 2026-11-08 20:00 · L2 · demo-game frames at the decision points (for hero scene 3)
Decision: tools/probe_demo_decision.py writes probe_demo_decision.json. It is the same demo game (test 4,201) as probe_demo.json, read with the W and B+ probes at layer 12, so each frame shows the board where the model is about to choose a move. The pre-registered probe_demo.json stays as it is. The video should use the new file only with the "exploratory" label from the entry above.
Prediction: all 64 squares right in about half the frames (the test-set whole-board rate is about 50%). No frame below 58 of 64. No alternation between white and black frames.
Cost estimate: $0, CPU, about 4 minutes.
--- result (added after, 2026-11-08 20:02) ---
runs/L2-20261108-1509-spelled/probe/probe_demo_decision.json, 77 frames. All 64 squares right in 46 of 77 frames (60%). 60-63 of 64 in most of the rest; the worst frame is ply 65 with 54 of 64. No alternation between white and black frames.
Was I right? About half the frames perfect: close (60%). None below 58: wrong, one frame at 54 (ply 65, deep in the endgame). No alternation: right.

## 2026-11-08 20:02 · L2 · Stockfish sweep over the A/B checkpoints (camera spec: W/D/L and legality at every eval)
Decision: tools/sf_sweep.py on L2-20261108-1509-spelled at its saved checkpoints: steps 0, 50, 100, 200, 500, 1000, 2614, 4227, 5840, 7454, 9068. Each plays 20 games against Stockfish 19 Level 0 at 0.1 s, same protocol as the match, with no extra games for the move count. Step 10,681 is not replayed: its row is the match itself (games/final_sf0), because replaying the match checkpoint would be a second roll of the same dice. Output: runs/L2-20261108-1509-spelled/eval_stockfish.jsonl and games/sweep_sf0_step*/. This is descriptive logging for the chart. It is not part of the pass decision.
Prediction:
- Steps 0-200: every game forfeited within its first few moves. Legal rate about 2% at step 0, about 70% at step 200.
- Step 500: about 85% legal, most games forfeited.
- Step 1000: about 89% legal (about 3 points below its held-out 92.1%), no wins.
- First win at step 2614 or 4227.
- Steps 5840-9068: 96.5-97.5% legal, 2-5 wins each.
- In general, match legality runs about 1 point below the held-out rate at each checkpoint once training is past step 2000.
Cost estimate: $0, CPU, about 15 minutes.
--- result (added after, 2026-11-08 20:19) ---
First attempt crashed at step 0: the untrained model's output held a character (U+FFFD) that Windows' default file encoding can't write. tools/play_stockfish.py now writes all its files as UTF-8. Moves and results are unchanged; the match files written before were plain ASCII. I deleted the partial step-0 folder and reran the whole sweep. Console in runs/L2-20261108-1509-spelled/sf_sweep_stdout.log, rows in eval_stockfish.jsonl. I appended the match's 20 counted games as the step-10,681 row, labelled as such.
| step | W / D / L | forfeits | model moves | legal | held-out legal (nearest eval) |
|---|---|---|---|---|---|
| 0 | 0 / 0 / 20 | 20 | 26 | 0.0% | 2.2% |
| 50 | 0 / 0 / 20 | 20 | 109 | 49.5% | |
| 100 | 0 / 0 / 20 | 20 | 160 | 47.5% | |
| 200 | 0 / 0 / 20 | 19 | 266 | 74.8% | |
| 500 | 0 / 0 / 20 | 20 | 462 | 86.6% | |
| 1000 | 0 / 0 / 20 | 16 | 646 | 87.6% | 92.1% |
| 2614 | 0 / 1 / 19 | 9 | 662 | 93.1% | 96.3% (3k) |
| 4227 | 1 / 1 / 18 | 10 | 737 | 92.8% | 96.7% (4k) |
| 5840 | 5 / 3 / 12 | 6 | 754 | 96.2% | 97.9% (6k) |
| 7454 | 7 / 0 / 13 | 7 | 839 | 96.2% | 98.2% (7k) |
| 9068 | 7 / 5 / 8 | 4 | 862 | 97.8% | 98.6% (9k) |
| 10681 (the match) | 5 / 5 / 10 | 3 | 756 | 97.9% | 98.6% |
Was I right?
- Steps 0-200 all forfeited: right except one game at step 200 (19 of 20). About 2% legal at step 0: 0% over 26 moves. About 70% at step 200: 74.8%.
- Step 500 about 85%: 86.6%, all forfeited. Right.
- Step 1000 about 89% with no wins: 87.6%, no wins. Right.
- First win at step 2614 or 4227: 4227. Right.
- Steps 5840-9068 at 96.5-97.5% legal with 2-5 wins: legality 96.2, 96.2, 97.8, about right. Wins 5, 7, 7, more than I said.
- Match legality about 1 point below held-out after step 2000: wrong early (3-4 points at steps 2614-4227), right late (0.8 points at 9068, 0.7 at the end). The gap closes as training goes on.
What I learned: winning shows up before clean legality. The model wins its first game at step 4227 while still making an illegal first attempt in 7% of its moves, and wins 7 of 20 at step 7454 at 96% legal. Wins at the late checkpoints swing between 5 and 7 out of 20 from one 20-game match to the next. So "5 wins" in the official match is one draw from a wide spread, not a precise strength measure. Legality, the criterion that decides the level, moves slowly and steadily.

## 2026-11-08 22:31 · L2 · result: continuation run L2-20261108-1951-cont (and -cont-smoke)
--- result (added after) ---
Kernel kaggle-user/ep04-l2-cont, pushed 19:51:10. Session 8,741.5 s by the kernel's clock, logged as 2.449 GPU-h (start to COMPLETE seen at 22:18:11), $0. Week total 12.14 of 30 GPU-h. Both phases exit code 0, no crash, no NaN, no restart. 2 of 9,918 steps logged no grad norm (fp16 overflow steps skipped by the loss scaler, as designed).
Smoke phase, honest note: reading the 2.1 GB dataset from Kaggle's input disk took 16.5 min (the A/B's 1 GB read took 3.4 min, its smoke test 16 s). The smoke run's 8-minute limit counts from process start, so it stopped after step 1. It still did what it was for:
- DDP + compile + init_from + the new data ran a full step.
- Its step-0 eval repeated the old model: full-val loss 1.110437 at step 1 against the A/B's 1.110463.
- It wrote its checkpoints and final eval and exited 0.
But it was a 1-step smoke test, not the 60 steps I planned. The main phase read the file from cache in 18 s.
Actual (main run):
- 9,918 steps, 650.0M tokens. Median 95.7k tokens/s, peak memory 10.10 GB.
- The loader drew 1,269,504 windows = 0.53 of an epoch of the union. My 19:45 entry said 0.27, a wrong division. The new/old split (about 350M / 300M tokens) and "no window drawn twice" stand.
- Step-0 eval: val 1.1134, legal 98.76%, exactly the A/B's last 300-game eval.
- Train loss went from 1.095 (step 100) to 1.128 (steps 300-500) during the warm restart. Val loss was worse than the start at steps 1k-2k (1.1338, 1.1216) and better from step 3k on (1.1077).
- 300-game legality by eval: 98.76 (step 0), 98.59, 98.65, 98.80, 98.86, 99.01, 99.10, 99.17, 99.23, 99.21, 99.30%.
- Final eval (full val set, the same 2,000 test games):
| | A/B spelled-out (match model 1) | continuation (match model 2) |
|---|---|---|
| val loss | 1.1105 | 1.0339 |
| legal-move rate | 98.60% (about 2,066 illegal) | 99.23% (about 1,134 illegal) |
| exact SAN | 98.05% | 98.87% |
| human-move match | 49.12% | 51.46% |
| legal, plies 1-20 / 21-60 / 61+ | 99.97 / 98.67 / 97.06% | 99.99 / 99.27 / 98.39% |
Was I right?
- Step-0 eval repeats the old one: right, exactly.
- Warm-restart bump about 0.02, back below the old level by step 2,000: the bump was about 0.03 in train loss, and val loss was only back below the start at step 3,000. Wrong on timing.
- About 88k tokens/s and about 125 min: faster, 95.7k. The main phase took 126 min including evals.
- Final val 1.075 (range 1.055-1.095): wrong, 1.034 is better than my whole range.
- Legal 98.95% (range 98.7-99.3): 99.23%, inside the range and above my point.
- Human match about 50.3%: 51.5%.
- Cost about 2.35 GPU-h: 2.45 (the slow first data read).
What I learned: more game text roughly halved the late-game illegal rate (ply 61+: 2.94% to 1.61% illegal), and the model is past 99% on human games for the first time. The match decides: on the first model, the match ran 0.7 points below held-out. If that gap holds, the new model lands near 98.5% in the match, still short. The pre-registered match is next, unchanged: 20 games, first 1,000 model moves, final checkpoint step 9,918 only.

## 2026-11-08 22:31 · L2 · match 2 vs Stockfish level 0, checkpoint L2-20261108-1951-cont step 9918
Correction to the entry above: the continuation's final eval had 1,129 illegal first attempts, not "about 1,134" (rate 99.2332%).
Decision: the one new match fixed in the 19:13 and 19:45 entries. Same protocol as match 1, unchanged: Stockfish 19 Level 0, 0.1 s/move, 20 games, the model white in 1-10 and black in 11-20, greedy, 4 retries at T=1.0 then forfeit, 300-ply cap, extra games to 1,000 model moves (not counted for wins). Same tool, settings and seed (tools/play_stockfish.py, 6 workers, seed 7). The only code change since match 1 is UTF-8 file writing. Checkpoint runs/L2-20261108-1951-cont/ckpt/step_9918.pt, the final one. No other checkpoint will be tried. Output runs/L2-20261108-1951-cont/games/final_sf0/. This match settles Level 2 whatever happens.
Prediction: the 19:45 prediction stands as the pre-registered one (legality 98.6%, range 97.8-99.4; 6 wins, range 2-12). Updated now that I've seen the training result: legality 98.6% (held-out 99.23% minus the 0.6-0.7-point gap seen at the end of the first run). Chance of reaching 99%: about 30%. Wins 7 (range 3-12), at most 2 forfeits.
Cost estimate: $0, 0 GPU-h, about 2 minutes of CPU.

## 2026-11-08 22:33 · L2 · result: match 2, L2-20261108-1951-cont step 9918: FAIL on legality, 98.8%
--- result (added after) ---
Files: runs/L2-20261108-1951-cont/games/final_sf0/ (game_01..26.pgn, moves.jsonl, games.jsonl, summary.json, illegal_kinds.json); console in runs/L2-20261108-1951-cont/games_final_sf0_stdout.log. 26 games in 61 s of CPU, 0 GPU-h, $0.
Actual:
- The 20 match games: 12 wins, 3 draws, 5 losses, 0 forfeits.
  - As white (games 1-10): 7 wins, 2 draws, 1 loss.
  - As black (11-20): 5 wins, 1 draw, 4 losses.
- The 20 games held 878 model moves (10 illegal first attempts, 98.86%). 6 extra games made up the 1,000.
- First-attempt legality over the first 1,000 model moves: 988 of 1,000 = 98.8%. The bar is 990. Exact SAN 97.7%; 10 of the 11 legal-but-not-exact moves differ only in a + or # mark.
- Pass = at least 99% legal AND at least one win. Wins pass; legality fails by 2 moves. Level 2 is a FAIL for this checkpoint, and by the 19:13 rule there is no further try.
- The 12 illegal first attempts:
  - no_such_move 6
  - not_a_move 5: all five are "1-0", the model resigning as black, in games 18 and 22
  - ambiguous 1
- By stage: 0 in plies 1-20 (252 moves), 5 in 21-60 (425 moves, 1.2%), 7 at ply 61+ (323 moves, 2.2%).
- Compared with match 1 over the same 1,000-move window:
  - board errors (no_such_move + king_in_check + ambiguous) fell from 19 to 7;
  - resignations rose from 1 to 5.
  The model now loses more often by writing the result than by misreading the board.
Was I right?
- Legality 98.6% (range 97.8-99.4): 98.8%, close, and the expected FAIL came true. My about-30% chance of passing did not come up.
- Wins 6 (range 2-12), updated to 7 (range 3-12): actual 12, the top of both ranges. The extra training made it much stronger, not just more legal.
- At most 2 forfeits: 0. Right.
What I learned: the bar was missed by two moves out of a thousand. With about 12 expected misses, a single 1,000-move match has a spread of about ±3.5 moves (Poisson), so this model sits right at the line: another match could well land above 990. That is exactly why the protocol allowed one match per checkpoint and one extra try, and I'm keeping to it. Not counting resignations would give 993 of 995 = 99.8%, but that would be changing the rule after seeing the result, so it is not the verdict. It is a fair thing to show on screen as a finding: by now the model's biggest single source of "illegal moves" is that it has learned to resign.

## 2026-11-08 22:33 · L2 · corrections to the match-2 result entry, and the Stockfish sweep for the continuation
Corrections (the entry above is left as written):
- Colour split: as white (games 1-10) 6 wins, 2 draws, 2 losses; as black (11-20) 6 wins, 1 draw, 3 losses. Not 7/2/1 and 5/1/4. The 12 / 3 / 5 total is right.
- Without the resignations it would be 988 legal out of 995 real move attempts = 99.3% (or 993 of 1,000 = 99.3% if a resignation counted as legal), not "993 of 995 = 99.8%". The verdict doesn't change: it stays the pre-registered count, 98.8%.
Sweep (descriptive, for the camera spec's W/D/L at every eval): tools/sf_sweep.py on L2-20261108-1951-cont at steps 1000, 2486, 3973, 5459, 6945, 8432. That is 20 games each, same settings as before. Step 0 is the old model's step 10,681, already covered by match 1; step 9,918 is match 2 itself (not replayed; I'll append its 20 counted games as a labelled row).
Prediction: legality 97.5-98.5% at every step, rising slowly. Wins 5-10 per 20 games, tending to rise.
Cost: $0, CPU, about 8 minutes.
--- result (added after, 2026-11-08 22:38) ---
runs/L2-20261108-1951-cont/eval_stockfish.jsonl (sweep console in sf_sweep_stdout.log):
| step | W / D / L | forfeits | model moves | legal |
|---|---|---|---|---|
| 1000 | 5 / 3 / 12 | 5 | 778 | 96.5% |
| 2486 | 6 / 3 / 11 | 5 | 715 | 97.1% |
| 3973 | 10 / 6 / 4 | 1 | 934 | 98.8% |
| 5459 | 3 / 5 / 12 | 4 | 880 | 98.6% |
| 6945 | 7 / 1 / 12 | 5 | 741 | 98.9% |
| 8432 | 6 / 5 / 9 | 6 | 873 | 97.6% |
| 9918 (match 2) | 12 / 3 / 5 | 0 | 878 | 98.9% |
Was I right? Legality 97.5-98.5%, rising slowly: wrong in both parts. It ranges from 96.5% to 98.9%, and it isn't monotonic (8432 dips to 97.6%). Wins 5-10, rising: mostly in range (3 to 12), but not rising steadily.
What I learned: 20-game matches are noisy, in wins (3 to 12 across neighbouring checkpoints) and in legality (about ±0.5 points on 700-900 moves). The 300-game held-out legality climbs smoothly over the same checkpoints (98.6% to 99.3%). So for the chart, the held-out curve is the trend, and the match rows are single dice rolls around it.

## 2026-11-08 22:39 · L2 · VERDICT: Level 2 FAIL (legality 98.8% against the 99% bar; wins passed)
The pass rule (14:21 entry): at least 99% of first attempts legal over the first 1,000 model moves in match games, AND at least one win in 20 games against Stockfish 19 Level 0 at 0.1 s per move.
| | checkpoint | held-out legal (2,000 games) | match legal (first 1,000) | W / D / L (20 games) | verdict |
|---|---|---|---|---|---|
| match 1 | L2-20261108-1509-spelled step 10,681 (350M tokens) | 98.60% | 97.9% | 5 / 5 / 10 (3 forfeits) | FAIL (legality) |
| match 2, the one extra try | L2-20261108-1951-cont step 9,918 (+650M tokens) | 99.23% | 98.8% | 12 / 3 / 5 (0 forfeits) | FAIL (legality, 2 moves short) |
Level 2 is a FAIL. The model learned to play from text alone, well enough to beat Stockfish Level 0 twelve times in twenty. But it did not reach 99% legal moves in match play. Its last match missed by 2 moves out of 1,000, and 5 of its 12 misses were it writing "1-0" to resign.
Does it keep a board in its head? Yes, measured three ways:
- Pre-registered probe (last token of every move): 90.5% of squares and 81.2% of occupied squares, against 79.9% / 57.2% for the untrained network and 73.7% / 40.5% for the ply-only guess.
- Exploratory follow-up (where the next token is a move): 98.5% of squares, 97.7% of occupied squares, and the whole board exactly right about half the time. Where the next token is only a move number, the same board drops to 71% of occupied squares: it builds the board when it needs to move.
- The illegal moves show what's missing: no illegal first attempt in plies 1-20 in either match (536 moves). Most errors are late-game pieces moving through squares that aren't empty in reality.
My own idea (spelled-out move embeddings): works by the pre-registered rule, on one seed pair. Val loss 0.012 lower at equal steps and about 6% fewer illegal moves on 147k held-out positions; the gap was the same sign at every eval.
Skill probe: a weak signal only (correlation 0.35 trained vs 0.29 untrained).
Cost: Level 2 used 5.11 GPU-h of Kaggle quota (smoke 0.23, A/B 2.43, continuation 2.45), $0 of real money. Week total 12.14 of 30 GPU-h; 0 of 20 TPU-hours. All matches, probes and data preparation ran on this PC's CPU.
Human interventions in Level 2: none. Nothing was added to interventions.jsonl.
Rules check:
- My own code throughout: tokenizer, model, data pipeline, training loop, evaluation, match runner.
- python-chess only as referee and as the source of the true board for the probe labels. Stockfish only as the opponent.
- Training text is only Lichess games (CC0). Nothing I wrote or generated was trained on.
Was the plan right (14:21 entry)?
- Held-out legality 98.8 / 99.1%: the first run reached 98.50 / 98.60%; the continuation 99.23%.
- Match legality about 98.5%, the criterion "most likely to fail": right that it fails (97.9%, then 98.8%).
- 6 wins (2-12): 5, then 12.
- Probe best layer two thirds up at about 95%: wrong. It's the last layer, at 90.5% as pre-registered and 98.5% at the decision points.
- About 3 GPU-h for Level 2: wrong, 5.11, because the contingency run was needed.
Mistakes I made and logged in later entries (old entries left unedited):
- two header times written by hand (14:40 and 19:16);
- a wrong range for the 300-game legal gaps;
- "a rook down" (a queen);
- the demo-game index and frame claims;
- the repeat-token arithmetic and the epoch fraction of the continuation;
- the match-2 colour split and the no-resignation rate;
- a 1-step (not 60-step) smoke phase.
Next: Level 3. About 17.9 GPU-h plus 20 TPU-hours remain this week.

## 2026-11-08 22:39 · L2 · check of one claim in the verdict
The verdict says "most errors are late-game pieces moving through squares that aren't empty in reality". I checked this after writing it, splitting each match's no_such_move first attempts (first 1,000 moves) by cause, against the real board:
| | path blocked (piece could reach on an empty board) | no piece of that type in reach | pawn moves |
|---|---|---|---|
| match 1 | 10 | 2 | 1 |
| match 2 | 3 | 2 | 1 |
- A blocked path is the commonest single board error: 13 of the 19 no_such_move attempts. But it is not "most errors": it's 13 of all 33 illegal first attempts.
- Late-game share: 11 of 21 in match 1 and 7 of 12 in match 2 come after ply 60, from 22% and 32% of the moves.
Corrected line for the video: "Its commonest board mistake is sliding a piece through a square that is no longer empty, and its mistakes pile up late in long games."

## 2026-11-10 20:29 · L3 · plan (fixed before any Level 3 run)
Decision: Level 3 pretrains on FineWeb-Edu on the TPU v5e-8, in JAX with Optax (rule 2 allows them for a TPU Level 3). No Flax: the model is a plain JAX function over a dict of arrays, all mine.
Why the TPU: the 2x T4 left this week (17.9 GPU-h) is about 40 TFLOP/s in practice, so 17.9 h buys about 2.6e18 FLOPs. That trains a 124M model on about 3.5B tokens, which is not enough to beat GPT-2 (Karpathy-style 124M runs on FineWeb-Edu only draw level with it after about 10B tokens). A v5e-8 has 8 x 197 TFLOP/s of bf16 peak; at 30-40% utilisation, 8 hours buys 1.4-1.8e19 FLOPs, 5-7x the whole GPU budget.
Plan:
1. Scoreboard first: my own HellaSwag scorer (lm-eval-style prompt: activity label + ": " + context, endings joined with a space, per-ending log-likelihood divided by the ending's length in characters), run on GPT-2 124M's official weights through my own GPT-2 implementation and my own reading of GPT-2's tokenizer files. That number is the bar. Same script, same item preparation for my model.
2. Data: HuggingFaceFW/fineweb-edu, the sample-10BT subset (about 9.7M documents, about 10B GPT-2 tokens). Tokenized by my own BPE inside Kaggle CPU-only kernels (internet on, no GPU/TPU quota used), saved as kernel outputs and mounted into the TPU kernels. Nothing that big goes through this PC's 2.6 MB/s upload.
3. Tokenizer: my own byte-level BPE (Level 1 code), vocab 32,768 (256 bytes + end-of-text + 32,511 merges), trained on about 1 GB of FineWeb-Edu text. The Level 1 merge loop scans every pair each merge, which is too slow for 32k merges over millions of chunks, so it gets a heap with lazy deletion.
4. Model: decoder-only transformer as in Levels 1-2 (pre-norm RMSNorm, RoPE, SwiGLU, no biases, tied embeddings), plus QK-norm for stability at high learning rates, head size 128 (the TPU's matrix unit is 128 wide), context 1024. Size about 250-350M parameters, fixed after the TPU smoke test measures speed: roughly compute-optimal (about 20 tokens per parameter) for the hours I can afford.
5. Optimizer: Muon (orthogonalised momentum, Jordan 2024) on the hidden weight matrices, AdamW on embeddings and norms, with the Muon update scaled to AdamW's update size so both share one learning rate. It is a known technique, reported to need about 1/1.3-1/2 the tokens of AdamW. Because it is my own implementation, the ablation session also runs a plain-AdamW arm to check it actually helps here before the main run uses it.
6. Schedule: linear warmup, constant, then linear decay to zero over the last 20-30% (warmup-stable-decay), so a run cut short by the clock still has a usable checkpoint and the decay can be timed to the real end.
My own idea for this level, the "look-ahead head": a second, cheap output that at every 4th position predicts the bag of tokens still to come in the current sentence (positions t+2 to the sentence's end, at most 32 tokens, order ignored), through the same tied embedding matrix as the next-token head. Weight 0.1 next to the language-model loss; never used for scoring. Hypothesis: HellaSwag punishes a model for preferring a fluent ending that does not fit the scene. Making each position also forecast what the sentence will be about, not just its next token, should push the network (and the embeddings it shares with the output) toward content-level planning. It is related to multi-token prediction (Gloeckle et al. 2024) and to bag-of-words losses in dialogue models, but predicting the unordered rest of the sentence as a pretraining auxiliary is my design. Multi-token prediction is reported to hurt small models, so I expect a cost in next-token loss; the question is whether HellaSwag gains anyway.
A/B test (rule 4), before the main run, small config, same seed, same initial weights, same tokens: Muon baseline vs Muon + look-ahead head, plus the AdamW arm. Decision rule (fixed now): the main run uses the look-ahead head only if (a) its HellaSwag acc_norm is higher than the baseline's AND (b) its validation next-token loss is no worse than the baseline's + 0.002. The main run uses Muon only if the Muon baseline beats the AdamW arm on validation loss.
Prediction:
- GPT-2 124M on my scorer: 31.1% (range 30.6-31.6).
- Look-ahead head at the small config: next-token loss +0.004 (worse), HellaSwag within noise of the baseline, so most likely NOT adopted (60%).
- Muon beats AdamW at equal tokens by 0.03-0.08 nats.
- Final model: HellaSwag acc_norm 34% (range 32-37%), beating GPT-2 by about 3 points.
Cost estimate: $0. TPU about 14-16 of 20 h (smoke 0.5, ablation 1.5-2, main run 9-11, evals). GPU 0-1 h. CPU kernels for tokenization (no quota). Week totals start at 12.14 of 30 GPU-h and 0 of 20 TPU-h.

## 2026-11-10 20:34 · L3 · L3-20261110-2034-gpt2-baseline (the bar)
Decision: Score GPT-2 124M (OpenAI's official weights, the openai-community/gpt2 model.safetensors file) on all 10,042 HellaSwag validation items with src/hellaswag.py, through my own GPT-2 code (src/gpt2_ref.py: safetensors reader, GPT-2 BPE from its vocab.json/merges.txt, the network). CPU on this PC, fp32. Two passes: (1) the harness convention, no prefix token, which is how the published 31.1% was measured; (2) the same with GPT-2's <|endoftext|> prepended, because my own model will be scored with its end-of-text token in front (it starts every document after one), and GPT-2 should get the same courtesy.
The bar (fixed now): the higher of the two acc_norm numbers.
Checks already done: "Hello world" encodes to [15496, 995], GPT-2's known ids; encode/decode round-trips. HellaSwag's text has 124 occurrences of the fractions ½ ¼ ¾, which Python's re would class as letters where GPT-2's pre-tokenizer says numbers, so the pre-tokenizer's letter and number classes are built from Unicode categories instead of \w. A 200-item trial took 20 s, so a full pass is about 17 min.
Prediction: pass 1 acc_norm 31.1% (range 30.6-31.6%), acc about 28.9%. Pass 2 within 0.5 points of pass 1.
Cost: $0, CPU only.

## 2026-11-10 20:36 · L3 · TPU environment probe (kernel ep04-l3-tpu-probe)
Decision: First TPU session, about 5 minutes: record the Kaggle TPU v5e-8 image (JAX/Optax/libtpu versions, devices, host CPUs, RAM, disk), measure bf16 matmul speed on one chip and on all 8, and time a 540 MB FineWeb-Edu shard download from Hugging Face inside Kaggle plus its parquet read.
Why: everything in the Level 3 plan depends on these numbers: whether JAX is preinstalled, how fast the chips really are, and whether tokenising inside Kaggle (CPU kernels or the TPU host) beats uploading from this PC.
Prediction: JAX preinstalled with TPU support, 8 devices of kind "TPU v5 lite". Matmul 8192x8192 bf16: about 150 TFLOP/s per chip (76% of the 197 peak), about 1,200 on all 8. Host: 100+ CPUs. Download 100+ MB/s.
Cost estimate: about 0.1 TPU-h, $0.

## 2026-11-10 20:45 · L3 · tokenizer (result) and data preparation (decision)
Tokenizer (result): own byte-level BPE, vocab 32,768 = 256 bytes + <|endoftext|> (id 256) + 32,511 merges, Level 1 pre-tokenizer. Chunk counts from the first 250,000 documents of sample/10BT/012_00000.parquet (1.180 GB of text, 1,480,738 distinct chunks, 238.6M chunks, 56 s on 8 processes); merges learned from the chunks seen at least twice, with the new heap trainer, in 259 s. Before that I checked the heap trainer against the Level 1 trainer on 3,000 documents: the same 800 merges with the same counts. Merge 1 is " " + "t", merge 6 is " the". On 20,000 held-out documents (skipping the first 300,000): 4.514 bytes per token, round-trip exact on 500 documents, 449k tokens/s on one process. Files: tokenizer/fineweb/ (tokenizer.json, merges.txt in learned order, train_log.txt, stats.json).
Was the plan's cost right? Yes, minutes on this PC.
Data preparation (decision): the 14 parquet files of sample/10BT (28.5 GB). Files 000-012 are training data (about 9.5B of my tokens), file 013 (the last, 0.54 GB, about 190M tokens) is validation only: no document in it is trained on. Tokenisation runs in four Kaggle CPU-only kernels with internet (no accelerator, so no quota), each downloading its parquet files from Hugging Face and writing uint16 token files (each document followed by <|endoftext|>) to its output: A = 000-003, B = 004-007, C = 008-011, D = 012 + 013 (val). The TPU kernels mount those outputs directly. The tokenizer and the HellaSwag file go up as a small private dataset (kaggle-user/ep04-l3-meta).
Why not this PC: it downloads at 62 MB/s but uploads to Kaggle at about 2.6 MB/s (Level 1), so 20 GB of tokens would take over 2 hours to upload.
Prediction: each CPU kernel encodes at about 1.2M tokens/s (4 CPUs) and finishes 4 files in about 45 min. Total about 9.7B training tokens and 190M validation tokens, 4.5 bytes per token.
Cost: $0, no accelerator quota.
Note on two earlier headers: "20:34 · gpt2-baseline" was appended at 20:33:45 and "20:36 · TPU environment probe" at about 20:35:30, both typed ahead of the clock again. From this entry on the header time is inserted by the shell's date command.

## 2026-11-10 21:37 · L3 · result: GPT-2 baseline, the bar is 31.27%
--- result: L3-20261110-2034-gpt2-baseline (both passes) ---
Actual: pass 1 (no prefix, the harness convention): acc_norm 0.311392 (se 0.0046), acc 0.289185, token-normalised 0.309301; in-domain 0.3075, zero-shot 0.3152; 1,879 s on CPU. Pass 2 (GPT-2's <|endoftext|> in front): acc_norm 0.312687 (se 0.0046), acc 0.290480, token-normalised 0.310197; 1,718 s. Same scorer (sha 7ebb7b8f4e53725b), same data file (sha 58efa63005494025), same weights (sha 248dfc3911869ec4). Files: runs/L3-20261110-2034-gpt2-baseline/baseline_gpt2.json and baseline_gpt2_prefix_eot.json, with per-item records and 5 example items each.
THE BAR: 31.27% acc_norm (pass 2, the higher one, as fixed before the passes). My model has to score above 0.312687 on all 10,042 items with this scorer.
Was I right? Yes. Pass 1 predicted 31.1% (range 30.6-31.6), got 31.14%; acc predicted about 28.9%, got 28.92%. Both match the published lm-eval numbers for GPT-2 (31.1 / 28.9), so the scorer reads HellaSwag the way the harness does. Pass 2 predicted within 0.5 points of pass 1: +0.13 points.
What I'll change: nothing. The 0.13-point gap is inside one standard error, but the rule said the higher one, so the bar is 31.27%.

## 2026-11-10 21:37 · L3 · result: tokenisation (10.02B training tokens ready on Kaggle)
--- result: tokenisation kernels ep04-l3-tok-a/b/c/d ---
Actual: all four COMPLETE (Kaggle CPU kernels, 4 CPUs each, no accelerator quota). 13 training files fwedu_000-012: 10,021,792,044 tokens (9.49M documents; each file about 770M tokens). Validation fwedu_013: 193,638,851 tokens (182,101 documents), never trained on. 4.49-4.53 bytes per token on every file (4.51 on the held-out tokenizer check). Encode speed 2.2-3.7M tokens/s per kernel; downloads from Hugging Face 5-42 s per 2.2 GB file. Wall time per kernel 8-23 min. Metadata: kaggle/out/tok/<a-d>/tok_meta_<a-d>.json (only the metadata and logs were downloaded; the token files stay on Kaggle and are mounted into the TPU kernels).
Was I right? Totals yes (predicted about 9.7B train and 190M val, got 10.02B and 194M). Speed no: predicted 1.2M tokens/s and about 45 min per kernel, got 2-3x faster (8-23 min). I assumed one process per CPU at my PC's single-process rate (449k/s), but the Kaggle CPUs ran faster than that and the download was not a bottleneck.
What I'll change: nothing for Level 3; the data is ready. 10.02B training tokens bounds the main run: at about 250k tokens/s, 8 hours is 7.2B tokens, so the main run will not repeat any data.

## 2026-11-10 21:37 · L3 · session 1: A/B test (3 arms) + main-config smoke tests (kernel ep04-l3-s1)
Decision: First real TPU session (kernel ep04-l3-s1, one session, runs back to back). Part 1 is the A/B test of the plan, three arms. Part 2 is three short smoke tests of the main-run candidate. The probe (20:36 entry) has been queued for over an hour, so this job does not wait for it. The kernel prints the same facts the probe would (JAX version, devices, CPUs, RAM, disk). If the image's JAX cannot see the TPU, it installs jax[tpu] first and logs that in session.json.
Change from the plan (20:29): no Optax. The optimizer is now all mine (src/l3_model.py: Optimizer = AdamW + Muon written out in JAX, Schedule = warmup-stable-decay computed on the host and fed to the step as a number). Why: the learning rate has to come from the host so the main run can place its decay from the measured speed and finish it before Kaggle ends the session ("time mode": flat until 75% of the time budget, then the decay length is fixed from the measured steps/s; predecay.npz is saved at that moment so the run can be extended later). Writing AdamW myself took a few lines once the schedule was on the host. Checked on CPU (tiny model): Muon + look-ahead, AdamW, and time mode all run; time mode ended on budget with the learning rate decayed to near zero. A run crashed on purpose at step 17 and resumed from latest.npz reproduced the uninterrupted run exactly: all 30 steps' train loss, look-ahead loss and learning rate identical, final val loss 7.767775 in both. A local dry run of the generated kernel on fake inputs also passed. It caught one bug before any TPU time was spent: the training glob would have included the validation file fwedu_013 (it sits in the same output folder as 012). The trainer's assert would have stopped it; the kernel now lists training files explicitly. Side effect of that dry run: its JAX-install fallback ran on this PC and replaced the local CPU JAX 0.12.0.1 with 0.4.35. I reinstalled 0.12.0.1, and the fallback is now disabled on dry runs.

Part 1, A/B arms (rule 4). Small config: 8 layers, d 512, 4 heads of 128, SwiGLU 1408, context 1024, vocab 32,768 = 42.5M parameters (25.7M non-embedding). Same seed 1337, so the same initial weights (the look-ahead projection comes from a separate key and does not disturb the others). Same data order (seed 2026) over training files 000-001 only (1.54B tokens, loaded in RAM), so no token is repeated. 3,800 steps of 256 x 1024 = 0.996B tokens. Peak lr 2e-3, warmup 200, flat, linear decay to 0 over the last 25%, weight decay 0.1, clip 1.0. Evals every 180 steps (22 during the run plus the final): val loss on the first 2M tokens of fwedu_013, and all 10,042 HellaSwag items. Final eval on 20M val tokens.
  L3-20261110-2137-abl-muon: Muon on the hidden matrices, AdamW on embeddings and norms. Baseline.
  L3-20261110-2137-abl-muon-la: the same plus the look-ahead head (weight 0.1, every 4th position, window 32).
  L3-20261110-2137-abl-adamw: AdamW on everything.
  Decision rules, unchanged from 20:29. Look-ahead adopted only if (a) final HellaSwag acc_norm > baseline's AND (b) final val loss <= baseline + 0.002. Muon kept only if abl-muon's final val loss < abl-adamw's.
Part 2, smoke tests of the main candidate: 24 layers, d 1024, 8 heads of 128, SwiGLU 2816 = 342M parameters. Muon, peak lr 1.5e-3, 524,288 tokens per step, all 13 training files memory-mapped.
  smoke-main: 16 sequences per chip x 4 accumulation, XLA attention, 6 min, latest.npz every 3 min, full HellaSwag eval at the start and end.
  smoke-main-m64: 8 per chip x 8, 3 min, fallback if 16 per chip does not fit.
  smoke-main-flash: as smoke-main with JAX's Pallas flash-attention kernel, 3 min.
  What I take from them: memory, tokens/s, eval time, and whether checkpointing a 342M model works. The main run's size is then the largest of {342M, about 240M (20 x 1024), about 160M (12 x 1024)} that gets at least 15 tokens per parameter in about 7.5 h of training at the measured speed.
Prediction:
- abl-muon final val loss 3.45 nats per token (range 3.30-3.60), about 1.10 bits per byte. HellaSwag acc_norm 27.5% (range 26.0-29.5): a 42M model on 1B tokens should not beat GPT-2.
- abl-muon-la: val loss +0.004 vs abl-muon (range -0.005 to +0.015); HellaSwag within +-0.8 points of the baseline. Most likely not adopted (60%), as in the plan.
- abl-adamw: val loss 0.03-0.08 worse than abl-muon, so Muon is kept (80%).
- Small-arm speed about 1.3M tokens/s (range 0.6-2.0M), about 13 min of training per arm plus about 5 min of evals.
- smoke-main: 16 per chip fits; about 235k tokens/s (range 150-350k), i.e. about 35% of the chips' bf16 peak counting 6N + attention FLOPs per token. m64 about 5% slower. Flash 5-15% faster than XLA attention at 1024 context (25% chance it fails to compile on this image). Full HellaSwag eval at 342M under 2 min.
Cost estimate: about 1.5 TPU-h (range 1.2-2.0) of the 20, $0. Session timeout set to 3 h.

## 2026-11-10 21:39 · L3 · result: TPU probe cancelled before it ran (one TPU slot per account)
--- result: TPU environment probe (kernel ep04-l3-tpu-probe), never ran ---
Actual: pushed at about 20:37, still QUEUED at 21:38. Pushing session 1 then failed with "Maximum batch TPU session count of 1 reached": a queued TPU kernel holds the account's only TPU slot. I deleted the never-started probe kernel (its code is in kaggle/build/ep04-l3-tpu-probe/) and pushed ep04-l3-s1 (version 1, 21:38). 0 TPU-h used.
Was I right? Untested: none of its predictions (JAX preinstalled, about 150 TFLOP/s per chip, 100+ CPUs, 100+ MB/s download) were measured. Session 1 logs the JAX version, devices, CPUs, RAM and disk, and measures real training speed, which matters more than a matmul benchmark. The download question is moot because the data is already tokenised on Kaggle.
What I'll change: one TPU job at a time, and every TPU job does real work, because each queue wait has cost about an hour so far. I did not plan for the one-slot limit.

## 2026-11-10 21:41 · L3 · session 1 cancelled after 80 s and re-pushed (checkpoint cleanup for smoke runs)
Decision: I cancelled session 1 about 80 s after it started (pushed 21:38:46, deleted 21:40:06; nothing ran past setup) and am re-pushing the same job, same run IDs, with one change. After each smoke run the kernel deletes that run's checkpoint files and logs every deleted file and its size in session.json and the run's events.jsonl.
Why: I added up the outputs after pushing. Each 342M smoke run writes latest.npz (weights + optimizer state, about 3 GB), a final fp32 file (1.4 GB) and fp16 checkpoints (0.7 GB each): about 16 GB for the three smoke runs. With about 4 GB from the A/B arms that is at or over the 20 GB Kaggle keeps from /kaggle/working, which could have lost the A/B logs. The smoke tests still save their checkpoints, so saving at 342M is still tested; the files are just not kept. The A/B arms keep everything (about 1.5 GB each). Tested with a local dry run of the kernel.
Prediction: unchanged from the 21:37 entry. The queue was instant last time; I expect a start within 10 min.
Cost: the cancelled session used at most 0.025 TPU-h (logged in the ledger).

## 2026-11-13 01:24 · L3 · session 1 queued 52 h without starting; deleted and re-pushed as ep04-l3-s1b
Decision: The re-pushed session 1 (kernel ep04-l3-s1, pushed 2026-11-10 21:41) sat QUEUED for about 52 hours and never started. At the manager's request I deleted it (01:23) and pushed the same job as a new kernel, ep04-l3-s1b, with run IDs re-stamped from the clock: L3-20261113-0123-abl-muon, -abl-muon-la, -abl-adamw, -smoke-main, -smoke-main-m64, -smoke-main-flash. They replace the L3-20261110-2137-* IDs, which never ran (the old job file is kept as kaggle/jobs/l3_s1_20261110-2137.json). A new kernel name is used in case the old name, deleted and re-created within three minutes on 11-10, was what stuck in the queue.
Why: no TPU time was used, so nothing is lost by restarting. The plan, settings, decision rules and predictions are unchanged from the 21:37 and 21:41 entries of 2026-11-10. The only code changes since are outside these runs: a --reset_schedule flag in the trainer (for extending a main run later) and the comparison tool tools/l3_compare.py.
Prediction: unchanged. Queue: unknown. The first push on 11-10 started within a minute; the second never started.
Cost: 0 TPU-h for the 52-hour queue (logged in the ledger). The TPU quota resets Saturday 2026-11-14 00:00 UTC.

## 2026-11-13 01:30 · L3 · session 1 moves to an interactive run in the Kaggle editor (manager presses Run)
Decision: The manager cancelled the queued batch run of ep04-l3-s1b on the Kaggle website (CANCEL_ACKNOWLEDGED at 01:29, about 5 min after the push, never started) and will run the same notebook interactively in the Kaggle editor, on TPU v5e-8, by pressing Run. The batch queue (what the CLI and "Save & Run All" use) has not started a TPU job of mine since 11-10. Nothing in the job changes: the same main.py, the same run IDs L3-20261113-0123-*, settings, decision rules and predictions as the 2026-11-10 21:37 entry. Two things differ for interactive runs. Files from an interactive session are not saved as notebook output, so the manager downloads /kaggle/working (runs/ and session.json) and puts it in kaggle/out/ep04-l3-s1b/. And the session's TPU time is taken from the kernel's own clock (session.json) plus the time until the manager stops the session.
Why: the 52-hour queue on the batch path. I cannot press Run in the editor, so this is a human intervention (logged).
Prediction: unchanged. One note: the queue_wait event the kernel writes counts from the 01:24 push, so it will include the time until the manager pressed Run.
Cost: unchanged, about 1.5 TPU-h, $0.

## 2026-11-13 01:36 · L3 · session 1 back to a batch run (ep04-l3-s1b version 3)
Decision: The interactive run did not happen. At 01:35 the manager asked me to run it again myself, so I pushed the same job from the CLI as a batch run: kernel ep04-l3-s1b version 3, pushed 01:36, QUEUED. Version 2 is a save from the web editor with no run and no output files (its log only renders the notebook). Same main.py, same run IDs L3-20261113-0123-*, same settings, rules and predictions as the 2026-11-10 21:37 entry. The intervention line of 01:30 records the manager's cancel of version 1; the interactive run that line mentions was dropped.
Prediction: unchanged.
Cost: about 1.5 TPU-h if it runs, $0.

## 2026-11-15 10:59 · L3 · result: session 1. Look-ahead head NOT adopted, Muon kept; 342M speed not measured (data-bound)
--- result: session 1 (ep04-l3-s1b version 3): A/B test + main-config smoke tests ---
Actual, the session: pushed 2026-11-13 01:36, started 2026-11-15 09:48 after 56.2 h in the batch queue, ran 57.9 min (0.99 TPU-h, the first TPU time of the week that began 11-14 00:00 UTC). Image: JAX 0.11.0 (preinstalled, saw the TPU, nothing installed), 8 x "TPU v5 lite", 224 CPUs, 396 GB RAM, 20.9 GB free in /kaggle/working. Outputs: runs/L3-20261113-0123-*, comparison in runs/L3-20261113-0123-ablation.json (tools/l3_compare.py).
Actual, the A/B arms (42.5M, 3,800 steps, 0.996B tokens each, all 23 evals logged):
  abl-muon (baseline): final val 3.2948 (1.0535 bits/byte), HellaSwag acc_norm 0.2837, acc 0.2722. 1.376M tokens/s, 14.1 min.
  abl-muon-la (look-ahead head): final val 3.2958 (+0.0011), HellaSwag 0.2825 (-0.12 points). Paired over all 10,042 items: -0.0012 +- 0.0027 (366 items right only with the head, 378 right only without). Look-ahead loss ended at 5.95 nats (it learns something about the bag of tokens to come). 1.198M tokens/s (-13%), 15.8 min.
  abl-adamw: final val 3.3286 (+0.0338 vs Muon), HellaSwag 0.2816. Paired, Muon minus AdamW: +0.0021 +- 0.0029. 1.398M tokens/s, 13.7 min.
  The curves stay in the same order from step 360 on: Muon below AdamW on val loss at every eval; the look-ahead arm within 0.002 of the baseline at every eval, sometimes above, sometimes below.
Decisions by the rules fixed on 11-10 (20:29 and 21:37 entries):
  Look-ahead head: NOT adopted. Rule (a) fails (HellaSwag 0.2825 is not above 0.2837); rule (b) passes (+0.0011 <= +0.002). The main run trains without it.
  Muon: KEPT (val 3.2948 < 3.3286).
Actual, the smoke tests of the 342M candidate (24 x 1024):
  smoke-main (16 seq/chip x 4): fits (no OOM), evals work (val on 2M tokens 4 s, full HellaSwag 18 s the first time with compilation, 4 s after), latest.npz saved at step 5 (2.9 GB). But only 9 steps in 6.3 min.
  smoke-main-m64 (8 seq/chip x 8): 14 steps in 3.3 min, peak memory 3.0 GB per chip of 16, and 32% of the first 10 steps' time spent waiting for data (data_wait_frac 0.32).
  Diagnosis: the smoke runs read all 13 training files memory-mapped from /kaggle/input, and random 2 KB windows from that mount are slow (one reader thread, 512 random reads per step). The A/B arms loaded their two files into RAM and waited for data 0.01% of the time. So the 342M compute speed was NOT measured: the smoke steps were data-bound, and their first window also includes compilation.
  smoke-main-flash: failed at compile. "Pallas TPU requires a libtpu version that's at most a month old"; the image's libtpu was built 2025-06-12. Flash attention is not available on this image, so XLA attention stays.
Was I right?
  Look-ahead: yes. Predicted +0.004 val loss (range -0.005 to +0.015), got +0.001; HellaSwag within noise, got -0.12 points; most likely not adopted (60%), not adopted.
  Muon vs AdamW: yes, AdamW 0.034 nats worse (predicted 0.03-0.08), at the low end.
  Small-arm absolute numbers: val loss better than my range (3.29 vs predicted 3.45, range 3.30-3.60; 1.05 bits/byte vs 1.10); HellaSwag inside it (28.4% vs 27.5%, range 26.0-29.5). Speed right (1.38M tokens/s vs about 1.3M), as was about 13 min of training per arm.
  Smoke: "16 per chip fits" yes; the speed prediction (235k tokens/s) untested, because I did not foresee that the input mount is slow for random reads. Flash: I gave 25% to a compile failure; it failed.
  Queue: wrong by far. I expected a start within 10 min; it took 56 h.
What I'll change:
  1. The main run loads all 13 training files into RAM (20 GB of the 396 GB), as the arms did with 2. No memory-mapped reads from /kaggle/input.
  2. Because the 342M speed is still unknown and every TPU session costs days of queue, the main session must size itself. It first runs a short in-RAM speed test of the 342M model, then picks the model size by the rule fixed on 11-10 (the largest of 342M / 239M / 162M that reaches 15 tokens per parameter in the training budget), then trains it in time mode. Details and predictions in the next entry.
  3. Correction to the 11-10 21:37 entry: "about 240M (20 x 1024)" and "about 160M (12 x 1024)" had the wrong layer counts. 20 x 1024 is 290.5M and 12 x 1024 is 187.7M. The sizes I meant are 16 x 1024 = 239.1M and 10 x 1024 = 162.0M.
  4. tools/l3_compare.py first paired items by their "ind" field, which repeats across HellaSwag's splits (9,609 distinct of 10,042). It now pairs by file position (identical order in every run); the numbers above are from the fixed version.

## 2026-11-15 11:04 · L3 · session 2: speed test, self-sized main run (kernel ep04-l3-main)
Decision: Session 2 (kernel ep04-l3-main, one TPU v5e-8 session, timeout 9 h), approved by the manager ("Yes, push it"). Two runs back to back:
  1. L3-20261115-1104-speed-342m: the 342M model (24 x 1024, 8 heads of 128, SwiGLU 2816), all 13 training files loaded into RAM (10.02B tokens, 20 GB), Muon, lr 1.5e-3, 16 sequences per chip x 4 accumulation = 524,288 tokens per step, XLA attention, 8 min from start (loading included). Metrics every 5 steps. Its checkpoints are saved, then deleted by the kernel.
  2. L3-20261115-1104-main. The kernel takes the median tokens/s of the speed test's metric records after step 10 (step 10 and earlier include compilation). It scales that speed to each candidate by FLOPs per token (6N + 12 x layers x 1024 x 1024). It then picks the largest of 342M / 239M (16 x 1024) / 162M (10 x 1024) that reaches 15 tokens per parameter in the 470-min training budget. That is the rule fixed on 11-10, written into the job before seeing any speed. For 342M the threshold is 181.9k tokens/s (5.13B tokens). The choice, the table behind it and the derived layout (plan steps, eval every plan/30 steps) go to runs/L3-20261115-1104-main/size_decision.json and a "decision" event.
  Main-run recipe: Muon on the hidden matrices + AdamW on embeddings and norms (session 1: Muon beat AdamW by 0.034), no look-ahead head (session 1: not adopted). Peak lr 1.5e-3 (2e-3 at width 512 in the A/B, scaled by 1/sqrt(2) for width 1024, because the Muon update is RMS-matched, which makes its spectral size grow with sqrt(width)). Warmup 300 steps, flat, then a linear decay to 0 over the last 25% of the 470-min budget, its length set from the measured steps/s (time mode, tested on CPU). predecay.npz is saved when the decay starts, so the run can be extended from there in another session. Weight decay 0.1, clip 1.0, seed 1337, data order seed 2026 (no token repeats: at most about 7B of the 10.02B tokens). Evals: val loss on 2M held-out tokens and all 10,042 HellaSwag items, about 30 times plus the final (20M val tokens). Samples at log-spaced steps, 10 fp16 checkpoints plus the final fp32 one and latest.npz with optimizer state every 28 min (rule 7).
Why: session 1 left the 342M speed unmeasured (data-bound smoke runs), and a TPU session has cost up to 56 h of queue. A separate speed session would cost days, so the session measures and decides for itself, by a rule fixed before the measurement.
Prediction:
- Speed test: 342M at 230k tokens/s (range 170-300k), about 34% of the bf16 peak counted as 6N + attention (the 42.5M arms got 27%). Data wait under 1%. Loading 20 GB into RAM under 4 min.
- Size: 342M chosen (65%), else 239M.
- With 342M: about 6.5B tokens (about 19 per parameter). Final val loss 2.85 nats per token (range 2.75-2.98), about 0.91 bits per byte.
- HellaSwag at the end: 33.0% (range 30.5-36.0%). Chance it beats the bar of 31.27%: about 75%. This is lower than the plan's 34% because of a new data point: the 42.5M arm got only 28.4% after 1B tokens. Going from there to 342M on 6.5B tokens is about 50x the compute; at the slope from that arm to GPT-2's 31.3% that adds only about 3 points, and the slope usually gets steeper as models get bigger. HellaSwag climbs slowly during the flat phase (about 31% when the decay starts) and gains 1.5-2 points during the decay.
Cost estimate: about 8.2 TPU-h (setup 1 min, speed test 10, loading and compilation 5, training 470, final eval and saves 5 min), $0. The week so far: 1.015 of 20 TPU-h. If the run ends below the bar, the fallback is a second session that continues from predecay.npz with more flat-phase tokens, then decays again (about 10 TPU-h left this week).

## 2026-11-17 17:50 · L3 · result: session 2. The 342M model scores 40.69% on HellaSwag; GPT-2's bar is 31.27%: Level 3 beaten
--- result: session 2 (ep04-l3-main version 1): speed test + self-sized main run ---
Actual, the session: pushed 2026-11-15 11:05, started 2026-11-17 05:04 after 42.0 h in the batch queue, ran 8.07 h (8.09 TPU-h with the teardown allowance; the week so far 9.08 of 20). Same image as session 1 (JAX 0.11.0, 8 x TPU v5 lite, 396 GB RAM). Outputs: runs/L3-20261115-1104-speed-342m, runs/L3-20261115-1104-main (logs, size_decision.json, session.json, the kernel log as kernel_log.json; checkpoints downloaded separately).
Actual, the speed test: 342M with all 13 training files in RAM, 125 steps in 8.1 min. 219.6k tokens/s in every metric record after step 10 (23 records, 219,548-219,576; the first one, 190k, includes compilation). From session start to the run's first line (setup, JAX check, loading 20 GB into RAM): 3.1 min. Data wait 0.
Actual, the decision (rule fixed on 11-10, applied by the kernel): at 219,569 tokens/s the 470-min budget gives 342M 6.19B tokens = 18.1 per parameter, so 15 is reached; 342M chosen. Plan 11,810 steps, eval every 393 (size_decision.json).
Actual, the main run: 341.9M parameters, 11,676 steps, 6.12B tokens (17.9 per parameter), 219.7k tokens/s throughout (32.8% of the v5e-8's bf16 peak, 8 x 197 TFLOP/s, counted as 6N + attention), data wait 0.00%, peak memory 3.0 GB per chip. The decay started at step 8,760 (4.59B tokens, 352.8 min in); predecay.npz saved. latest.npz saved 16 times (every 28 min), 10 fp16 checkpoints, final fp32 with optimizer state.
  Final eval (20M val tokens, all 10,042 HellaSwag items): val loss 2.7027 nats per token, 0.8642 bits per byte. HellaSwag acc_norm 0.4069 (+- 0.0049), acc 0.3390, token-normalised 0.4023.
  The curve (31 evals): 26.4% at step 0, above GPT-2's 31.27% from step 1,572 on (0.82B tokens), 37.8% at the last eval before the decay (step 8,646), 40.8% at step 11,397, 40.69% at the end. The decay added 2.9 points and took val loss from 2.830 to 2.703.
  Kernel exit: the training process printed "done" at 07:04:46 after writing every file, then hung in the JAX/TPU teardown and was killed (SIGTRAP, core dump): exit code 2, about 3 min lost. It was the last run, so stop_on_fail skipped nothing.
VERDICT, Level 3 scoreboard (HellaSwag validation, all 10,042 items, zero-shot, log-likelihood normalised by ending length in characters; src/hellaswag.py builds, scores and aggregates the rows for both models; data file sha 58efa63005494025 for both): my model 40.69%, GPT-2 124M 31.27% (the bar, with the EOT prefix; 31.14% without). Paired over items: +9.42 points +- 0.39 (1,266 items right only for mine, 320 right only for GPT-2). LEVEL 3: BEATEN.
Checks before believing it: the two item files list the same items, labels and ending lengths in the same order. The per-ending log-likelihoods are the same size as GPT-2's (mean per character on the correct ending -0.636 vs GPT-2's -0.684, wrong endings -0.707 vs -0.721), so the model is not seeing the answer tokens. The curve rises smoothly, with no jump.
Where the gain comes from: WikiHow items (6,799) 27.50% -> 39.11%, +11.6 +- 0.5 points; ActivityNet items (3,243) 39.16% -> 44.00%, +4.8 +- 0.7. This run cannot separate two explanations: FineWeb-Edu is full of how-to text (WikiHow's register), and some WikiHow pages may be in the training data (contamination). On the ActivityNet items alone the model still beats GPT-2, by 7 standard errors. Contamination is not measured yet.
Was I right?
  Speed: yes. 219.6k tokens/s vs 230k (range 170-300k); 32.8% of peak vs 34%. Data wait under 1%: yes (0). Loading under 4 min: yes (about 3).
  Size: 342M, as I expected (65%).
  Tokens: 6.12B vs about 6.5B. My 6.5B assumed 230k tokens/s; at the measured speed the budget gives 6.19B, and evals and saves took the rest.
  Val loss: better than my range. 2.70 vs 2.85 (range 2.75-2.98); 0.864 bits per byte vs 0.91.
  HellaSwag: wrong by far, in the good direction. 40.7% vs 33.0%, outside my range (30.5-36.0). "Beats the bar" (75%): yes. "About 31% when the decay starts": 37.8%. "+1.5-2 points in the decay": +2.9.
  Why I was wrong: I drew a line from the 42.5M arm to GPT-2's score as if GPT-2 sat on my models' curve. It does not. At equal tokens (1.0B), the 342M model had 31.9% where the 42.5M arm had 28.4%, and in the flat phase it gained about 1.7 points per billion tokens. GPT-2 was trained on WebText with a 2019 recipe, so it is not a point on my curve. The WikiHow gap above is part of it too.
  Cost: 8.09 TPU-h vs 8.2. Right. Queue: 42 h (session 1: 56 h).
What I'll change:
  1. Level 3 is beaten by the fixed scoreboard, so no extension from predecay.npz. The rest of this week's TPU quota (about 10.9 h) is not needed for the bar.
  2. Before the result goes into the video, measure contamination: a CPU-only scan of the 10.02B training tokens for 13-token n-grams of the 10,042 HellaSwag items (a Kaggle CPU kernel, no GPU or TPU quota), then score both models on the items with no overlap. If mine still beats GPT-2 on the clean items, the result stands without an asterisk. This is a new kernel, so it waits for the manager's go-ahead.
  3. If there is another TPU job: train_l3.py ends with os._exit(0) after the final files are flushed, so the TPU teardown cannot hang, and l3_job.py treats a "done" event followed by a non-zero exit as a finished run.
  4. Forecasts: anchor HellaSwag predictions on my own measured curves (two model sizes now), not on another model's score.

## 2026-11-17 23:29 · L3 · contamination scan (run L3-20261117-2329-contam)
Decision: the manager said go ("Yes, run the scan"). One Kaggle CPU session (kernel ep04-l3-contam, no accelerator, 3 h cap) runs src/contam.py over the 13 training files the main run read (fwedu_000-012: 10,021,792,044 tokens, 9,786,899 windows; the kernel checks both numbers against the main run's config). Then, here on my PC, the per-item records the fixed scorer already wrote for both models (runs/L3-20261115-1104-main/hellaswag_items.jsonl, runs/L3-20261110-2034-gpt2-baseline/baseline_gpt2_prefix_eot_items.jsonl, the 31.27% bar) are split into clean and dirty items. No model is re-run and nothing in the scorer changes.
The test, fixed now, before any number exists:
  - Windows: every HellaSwag validation item as ctx_a + " " + ctx_b followed by each ending, cut at WikiHow's [..] markers, encoded with my BPE (as is and with a leading space), cut into 13-token windows. 231,591 context windows, 233,362 gold-ending windows (including those that run from the context into the gold ending), 687,230 wrong-ending windows; 1,133,074 distinct. 2,549 wrong-ending windows that are also context or gold windows (e.g. gold and wrong ending start with the same word) count only as context/gold.
  - Search: every 13-token window of the training files is hashed (64-bit) and looked up, and every hit is checked token by token. Tested on local data with planted items: the scanner found exactly the 130 occurrences a brute-force search found, with the same "seen" flags (tools/test_contam.py).
  - DIRTY item = any of its context or gold windows occurs anywhere in the 10.02B training tokens. CLEAN = the rest. This is the deciding split.
  - Also reported: dirty_seen (an occurrence lies inside one of the 5,978,112 training sequences the main run actually used, rebuilt from the loader's seeded order; sha of that order 6640fd5d66eb3abf under numpy 2.3.5 and under 2.5.4, the TPU image's version); heavy (at least half of the gold windows occur: the correct ending is in the data verbatim); leak (a wrong-ending window occurs; the wrong endings were machine-written for the dataset, so only a page that prints HellaSwag itself can contain them).
  - Verdict rule: on the CLEAN items, the paired difference in acc_norm (mine minus GPT-2) with its standard error. If it is positive with z > 3, the Level 3 win stands without an asterisk, and the video gives the clean-item numbers next to the full ones. If not, the video reports the win with the contamination caveat and the clean number. The scoreboard result (all 10,042 items, 40.69% vs 31.27%) does not change either way.
Why: the main run's lead was concentrated on WikiHow items (+11.6 points vs +4.8 on ActivityNet). FineWeb-Edu is crawled web text and WikiHow pages are on the web, so part of that gap could be the model having read the very articles HellaSwag's WikiHow items were cut from. A 13-token overlap test (the GPT-3 paper's method, there with GPT-2 tokens) separates the two.
Prediction:
  - Dirty items: 7% of all (range 2-20%). WikiHow 9% (range 3-25%), ActivityNet 2% (range 0.5-6%); most ActivityNet hits will be stock phrases, not copied captions. (A 3M-token local FineWeb-Edu sample matched no item, which says little at that size.)
  - Of the dirty items, about 65% have an occurrence inside the sequences the model trained on (61% of the windows were trained on, and some items occur more than once).
  - Heavy items: about a third of the dirty ones. Leak items: at most 10.
  - Clean items: paired gain +9.0 points (range +8.0 to +9.6), z above 15; my model beats GPT-2 on them with about 99% probability. The WikiHow clean gain stays near +10, about twice ActivityNet's: most of the WikiHow excess is the domain (FineWeb-Edu is full of how-to text), not copying (70%).
  - Dirty items: a larger gain, about +14 points (range +8 to +30); with few dirty items its error bar will be wide.
Cost estimate: one Kaggle CPU session of about 30 min (reading 20 GB of input files, then about 150 chunks of 67M windows on 4 cores at 3-6 s each). CPU sessions do not draw on the GPU or TPU quota: 0 GPU-h, 0 TPU-h, $0.

## 2026-11-17 23:38 · L3 · result: contamination scan. 211 of 10,042 items (2.1%, all WikiHow) have text in my training data; on the 9,831 clean items my model still leads GPT-2 by 9.2 points: the win stands
Actual, the session: kernel ep04-l3-contam version 1, pushed 23:28:35, started 9 s later on a Kaggle CPU machine (4 cores, 32 GB RAM), 325 s in all. Exit 0. 0 GPU-h, 0 TPU-h, $0. Checks inside the run: 13 files, 10,021,792,044 tokens and 9,786,899 windows, both as in the main run's config. The rebuilt training order has sha 6640fd5d66eb3abf under Kaggle's numpy 2.1.3, the same as under 2.3.5 and 2.5.4. Tokenizer sha bc3533abdd573457 and HellaSwag sha 58efa63005494025, as in the scorer runs. 0 hash collisions. Outputs: runs/L3-20261117-2329-contam (contam_items.jsonl, contam_summary.json, contam_examples.json, contam_occurrences.npz, contam_scores.json from tools/l3_contam_scores.py).
Actual, the scan: 10.02B windows in 304 s (32.9M windows/s). 10,660 occurrences of 4,134 distinct item windows, 6,421 of them inside sequences the model trained on. Found: context windows 1,916 of 231,591, gold 2,190 of 233,362, wrong 61 of 687,230.
  Dirty items: 211 (2.1%), all WikiHow: 211 of 6,799 WikiHow items (3.1%), 0 of 3,243 ActivityNet. dirty_seen 143, heavy 72, leak 37.
  What the found text is: wikiHow articles themselves, in FineWeb-Edu, in wikiHow's page layout ("Method 1 of 2:", "1Learn from witty people. The easiest way to improve your wit is ..."). 117 dirty items have 5 or more windows found (copies of the article); 94 have 1-4, mostly stock phrases wikiHow shares with other pages (the Poison Help number "1-800-222-1222" was found 1,839 times).
  The 37 "leak" items are not pages that print HellaSwag. All 61 wrong-ending windows found are stock phrases: a crisis hotline number, "1 to 2 inches (2.5 to 5.1 cm)", "Cream the butter and sugar until light and fluffy.". HellaSwag's wrong WikiHow endings were written by a model trained on WikiHow, so they reuse its stock phrases. Only 1 item has context, gold and wrong hits together, and those are stock phrases too. The premise of my leak label was wrong. Nothing found suggests HellaSwag itself is in the data.
VERDICT, by the rule fixed at 23:29 (clean items, paired diff > 0 with z > 3): STANDS.
  Clean items (9,831): mine 40.42%, GPT-2 31.22%, +9.21 +- 0.39 points, z 23.7. WikiHow clean +11.35 +- 0.47; ActivityNet (all clean) +4.84 +- 0.68.
  Dirty items (211): mine 53.08%, GPT-2 33.65%, +19.4 +- 2.9 points, a gain larger than on the clean items by 10.2 +- 2.9.
  The extra gain on dirty items does not come from training on their text. Dirty items whose found text lies inside a sequence the model trained on: +17.5 +- 3.3 (143 items); dirty items whose text it never trained on: +23.5 +- 5.6 (68). Among the 117 article copies: trained on +21.4 +- 4.8 (84), never trained on +21.2 +- 8.4 (33). GPT-2 also does better on the dirty items (33.65% vs 27.31% on clean WikiHow). These are popular, web-typical articles that a model trained on web text finds easier, whether or not it read them. (Caveat: "never trained on" counts a 13-gram that straddles two training sequences as not trained on.)
  Removing the dirty items moves the full-set lead from +9.42 to +9.21 and the WikiHow lead from +11.60 to +11.35. Contamination accounts for about 0.2 of the 9.4 points, and for almost none of the WikiHow-vs-ActivityNet gap (6.8 points before, 6.5 after).
Was I right?
  Dirty share: 2.1% vs 7% (range 2-20%), at the bottom of my range. WikiHow 3.1% vs 9% (3-25%): in range, low. ActivityNet 0% vs 2% (0.5-6%): below my range; I expected stock-phrase hits in the captions and there were none.
  Seen share of dirty items: 68% vs 65%. Right. Heavy: 34% of dirty vs about a third. Right.
  Leak: 37 vs at most 10. Wrong, and so was the premise behind it (above).
  Clean gain: +9.21 vs +9.0 (range 8.0-9.6), z 23.7 vs above 15. Right. "Most of the WikiHow excess is the domain" (70%): yes, nearly all of it.
  Dirty gain: +19.4 vs +14 (range 8-30). In range, above my point.
  Cost: 5.4 min vs about 30 min, and 0 quota as expected. The scan ran at 32.9M windows/s on 4 cores; my estimate assumed slow reads of the input files that did not happen.
What I'll change:
  1. The Level 3 result goes into the video with both numbers: 40.69% vs 31.27% on all items (the scoreboard), and 40.42% vs 31.22% on the 9,831 items with no 13-token overlap with my training data.
  2. A wrong-ending hit does not detect dataset pages for WikiHow items. Next time, call an item leaked only if hits cover all four endings, or match the dataset's own formatting ("[title]", "[step]").
  3. Keep the trained/not-trained split from the rebuilt data order in any contamination check: it is a free control, and here it is what shows the extra gain is not memorisation.

## 2026-11-18 00:49 · EP · plan (fixed before any Epilogue run)
Decision: teach the Level 3 model (L3-20261115-1104-main, final checkpoint ckpt/step_11676.npz, 341.9M parameters, 40.69% on HellaSwag) to chat by supervised fine-tuning on conversations written by people, then talk to it myself. Run IDs for this part start with LE- (level E, the Epilogue).
Data (rule 9: only text people wrote, no Claude output):
  - OpenAssistant/oasst2 (Apache-2.0): conversation trees written by volunteers, several replies per prompt, ranked by volunteers. The parquet export has no machine-written ("synthetic") messages. English only; deleted and failed-review messages dropped. Train split: 5,126 trees; validation split: 275 trees, never trained on.
  - databricks/databricks-dolly-15k (CC-BY-SA-3.0): 15,011 instruction/response pairs written by Databricks employees. User turn = instruction, plus the context paragraph if there is one. 296 held out for validation (seeded shuffle), 228 dropped for not fitting 1,024 tokens.
  - Rule 9 filter: any message that calls itself Claude or says it was made by Anthropic is dropped with everything below it. Found 1 (an OASST reply that says it was "developed by Anthropic, PBC"), 0 in Dolly. Limitation: a volunteer could have pasted a Claude reply without that sentence, and I cannot detect it. 241 OASST replies call themselves ChatGPT/OpenAI or say "as an AI language model". I kept them: they are not Claude output, and rewriting them would mean training on text I wrote.
  - Not used: HuggingFaceH4/no_robots (written by people, but CC-BY-NC), and every machine-generated chat set (UltraChat, SmolTalk and the like; some mix in Claude output).
Format (src/chat_format.py): three new tokens <|user|> <|assistant|> <|end|> (ids 32768-32770), a conversation is <|endoftext|> <|user|> ... <|end|> <|assistant|> ... <|end|> ... Targets are the assistant's text and its <|end|>; user text is context only. <|end|> starts as a copy of <|endoftext|> (the base model already knows when a text is over); <|user|> and <|assistant|> start as the mean embedding.
Where: Kaggle 2x T4 in PyTorch, not the TPU (a TPU session has waited 42-56 h in the queue; the T4 queue has been instant). So the Level 3 network is rewritten in PyTorch (src/chat_model.py, same layout as the JAX dict). Checked before this entry, on CPU in float32 against the JAX code with the real weights: log-probabilities agree to 4.8e-5; HellaSwag on the first 300 items 47.67% in both, same prediction on 99.7% of items (the TPU numbers were bfloat16); runs/LE-port-check.json. The largest activation is 5,515 (the SwiGLU product), inside float16's range (65,504), so the T4s train in float16 autocast with float32 weights and residual stream.
Code checks before this entry (not experiments): src/train_chat.py ran 12 steps on CPU with a 4.5M random model through every eval; a crash at step 7 and a resume from latest.pt reproduced every step's loss, grad norm and learning rate exactly.
My own idea for this part, "runners-up": the volunteers wrote several replies to each prompt and ranked them. The usual recipe keeps only the best path through each tree and throws the other replies away. Mine keeps every reply people wrote and weights its loss by its place among its siblings: 1, 1/2, 1/4, 1/8 (best, 2nd, 3rd, 4th and later). Hypothesis: there are only about 2M tokens of best-path conversation, so three passes over them overfit a 342M model; the runners-up are more human writing of the same kind, and the down-weighting keeps the model leaning toward the replies people preferred. It belongs to the reward-weighted fine-tuning family; using the volunteers' ranks to bring back OASST's discarded branches is my design.
A/B test (rule 4), two runs at once, one per T4, same start, same seed, the shared examples in the same passes (tools/prepare_chat.py, seed 2026):
  LE-<time>-base: 3 passes over Dolly train (14,487) + the OASST top threads (5,029 threads, 7,489 replies; 239 cut back to the last whole reply that fits, 96 too long). 13,254 windows of 1,024 tokens = 414 steps of 32 windows, 13.6M tokens, 8.8M target positions.
  LE-<time>-rank: the same + the 26,314 other replies once (each its own example, the conversation above it as context, a third of them per pass). 22,244 windows = 695 steps, 22.8M tokens. More data means more steps: the question is "same recipe plus the runners-up", at 1.68x the compute, not equal compute.
Recipe (both): AdamW (betas 0.9/0.95), peak lr 5e-5, warmup 20 steps, linear decay to 0, no weight decay, clip 1.0, 32 windows per step (4 per micro-batch x 8; halved automatically if 4 do not fit). Evals about 12 per run: masked loss on the held-out OASST top threads (269 threads, 409 replies) and held-out Dolly (296); preference accuracy on 1,034 held-out pairs (the best-ranked reply vs each lower-ranked sibling, log-likelihood per character, normalised as the HellaSwag scorer does); HellaSwag (all 10,042, src/hellaswag.py) at start, middle and end. Samples at log-spaced steps: the 5 fixed camera prompts and 4 chat prompts, each as a user turn, greedy and T=0.8.
Decision rule (fixed now): the chat model is the rank arm only if, at the final eval, (a) its OASST val loss < base's - 0.01, (b) its pref_acc >= base's - 0.02, (c) its HellaSwag acc_norm >= base's - 0.01, and (d) its Dolly val loss <= base's + 0.02. Otherwise the base arm.
Smoke test first (rule 7): kernel ep04-chat-smoke, both arms for 30 steps, full HellaSwag at step 0 on the base arm (the port check, now on a T4 in float16), latest.pt and checkpoints saved and then deleted.
Prediction:
  - Port on the T4: HellaSwag at step 0 = 40.69% +- 0.15 (counting only the Level 3 vocabulary); with the chat tokens in the softmax, within 0.05 of that.
  - Speed 5.0k tokens/s per T4 (range 3.5-7k); 4 per micro-batch fits (60%).
  - Step 0, untuned: OASST val loss 2.75 nats per token (2.5-3.0); pref_acc 52% (48-56%).
  - Base arm, final: OASST val loss 2.20 (2.00-2.40); Dolly val loss 2.10 (1.85-2.40); pref_acc 55% (50-60%); HellaSwag 39.7% (38.0-40.8).
  - Rank arm minus base arm: OASST val loss -0.02 (-0.06 to +0.02); pref_acc -1 point (-4 to +2); HellaSwag -0.3 (-1.0 to +0.5); Dolly val loss +0.01. Rank arm adopted: 55%.
  - The chat afterwards: fluent and on topic for simple questions, ends its own turns, wrong on facts beyond common knowledge, and calls itself Open Assistant if asked who it is.
Cost estimate: $0. GPU: smoke about 0.35 h, main about 1.9 h (the rank arm about 80 min of training plus 20 min of evals and samples). This week (from 2026-11-14 00:00 UTC): 0 of 30 GPU-h used so far; TPU 9.08 of 20 h (separate quota, not needed here).

## 2026-11-18 00:53 · EP · LE-20261118-0053-before: the untuned model, asked the chat questions
Decision: before any fine-tune, put my seven opening chat questions to the Level 3 model as it is (ckpt/step_11676.npz, the three chat tokens added as in the plan, nothing trained), through the same chat program and settings the fine-tuned model will get: tools/chat.py, CPU float32, temperature 0.7, top-p 0.9, repetition penalty 1.1 over the last 64 tokens, at most 300 new tokens, seed 1000. One fresh conversation per question (runs/LE-20261118-0053-before/q1.json .. q7.json), so the later chat has a measured "before".
Questions: "Hi! Who are you?" / "What is the capital of France?" / "Why is the sky blue?" / "Can you give me three tips for sleeping better?" / "Write a short poem about the sea." / "What is 7 times 8?" / "Can an AI do AI research?"
Prediction: it has never seen <|user|> or <|assistant|>, so it continues the text as a web page would. At most 2 of 7 replies end their turn by themselves (<|end|> starts as a copy of <|endoftext|>, so it can end where a document would); 0 of 7 answer the question in the first sentence; at least 3 of 7 drift to an unrelated topic.
Cost: $0, CPU only, about 3 minutes.
--- result: LE-20261118-0053-before ---
Actual: 0 of 7 replies answer the question. 7 of 7 are web-page text about something else: the history of Tours (to "Who are you?"), female reproduction (to the capital of France), a Hungarian writer (to the blue sky), cancer and smoking (to sleep tips), tips for writing essays (to the poem), study habits (to 7 x 8), climate change (to whether an AI can do AI research). 3 of 7 ended their turn by themselves (200-251 tokens, at a point where a document could end); 4 ran to the 300-token cap. Four replies start with a fragment that is not English ("inazovanitrochonosestenie", "linde, können, görte"): <|assistant|> starts as the mean of all embeddings, which the model reads as the middle of some word. CPU speed 11.7 tokens/s, model load 1.2 s.
Was I right? Answers 0 and drift at least 3: yes (7 of 7 drifted). Turn endings: no, 3 against my "at most 2".
What I'll change: nothing; this is the "before". Files: runs/LE-20261118-0053-before/q1.json .. q7.json.

## 2026-11-18 01:27 · EP · result: LE-20261118-0050-smoke. Both arms ran out of GPU memory (base at step 16, rank at step 3); the port check passed
Actual: kernel ep04-chat-smoke version 1, pushed 00:50:59, started 10 s later on 2x T4, 1,107 s in all. Both runs exited with code 1, torch.OutOfMemoryError in the backward pass: the rank arm at step 3, the base arm at step 16 (the step after its step-15 eval). Peak allocated 14.73-14.83 GB at 4 windows per micro-batch, on a card with 15.6 GB (14.56 GiB). Step 1 peaked at 12.0 GB; from step 2 on, AdamW's state (2.7 GB, created at the end of step 1) is on top. The failed allocations (420-430 MiB) are the size of one float32 copy of the logits for about 3,400 target positions (3,400 x 32,771 x 4 bytes). The output layer ran on all of a micro-batch's target positions at once, so the peak depended on the data, and my automatic halving of the micro-batch only covered the first step.
  Port check (base arm, step 0, all 10,042 items, float16 on a T4): HellaSwag 40.749% with the chat tokens in the softmax, 40.759% counting only the Level 3 vocabulary, vs 40.69% from the JAX run in bfloat16. Rank arm, first 1,000 items: 46.4% vs 46.6% for Level 3 on the same items.
  Step 0 (untuned, both arms): OASST val loss 2.2054, Dolly 2.8118, pref_acc 64.3% +- 1.5. Base arm at step 15 (0.49M tokens): OASST 1.9562, Dolly 2.3382, pref_acc 62.8%. Training worked while it ran (no NaN, loss scale steady).
  Speed 2.83-3.05k tokens/s per T4. Samples took 101-117 s per round (18 generations one at a time), at steps 0, 1, 2, 5 and 10.
  Why pref_acc is so high before any tuning: the better-ranked reply is also the longer one in 69.8% of the 1,034 pairs (median 806 vs 438 characters), and log-likelihood per character favours longer texts. So pref_acc measures length as much as quality. It stays in the decision rule as fixed (it is only a guard there, rule b). From now on each eval also writes the per-pair scores (pref_scores.jsonl), so I can report a length-matched version as a secondary number.
Was I right?
  Port: yes. 40.76 vs 40.69 +- 0.15, and the chat tokens moved it by 0.01 (within 0.05).
  Speed: no. 3.0k vs 5.0k (3.5-7k), below my range.
  4 windows per micro-batch fits (60%): no.
  Step-0 OASST val loss: no. 2.21 vs 2.75 (2.5-3.0), below my range: a reply that follows a question is easier to predict than the FineWeb text the model scored 2.70 on.
  Step-0 pref_acc: no. 64.3% vs 52% (48-56); length, above.
What I'll change (all checked on CPU with the 4.5M test model before the next push):
  1. The output layer and loss run in chunks of 1,024 positions, each recomputed in the backward pass instead of kept (src/chat_model.py). Same loss to the last digit as the one-shot version; gradients equal to 1.6e-7 relative.
  2. The micro-batch fallback works at any step: drop the step's partial gradients, halve the micro-batch, double the accumulation (same windows, same loss), redo the step. With an artificial out-of-memory error at step 2 (--oom_test_step), every step's loss matched a run at the smaller micro-batch from the start (to 1e-6), and the final eval matched exactly.
  3. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in the kernel, and the CUDA cache is emptied after each eval and sample round.
  4. Disk: latest.pt is deleted just before it is rewritten (two runs x 4.1 GB plus temporary copies could reach the ~20 GB output limit). The final checkpoint is split into the float32 weights (step_<n>.pt, 1.4 GB) and the optimizer state (step_<n>_opt.pt, 2.7 GB), so I can download only the weights: this PC's C: drive has 5.5 GB free.
  5. Smoke tests sample only at the start and end (--sample_steps).
  6. The main run's time, redone at 3.0k tokens/s: the rank arm's 22.8M tokens are 2.1 h of training, not 80 min.
Cost: 0.32 GPU-h (1,107 s plus a 1-minute allowance), $0. This week: 0.32 of 30 GPU-h.

## 2026-11-18 01:27 · EP · LE-20261118-0128-smoke2: the smoke test again, with the memory fixes
Decision: before the main run (rule 7), re-run the smoke test with the fixes above, 25 steps per arm, on kernel ep04-chat-smoke2.
  - LE-20261118-0128-smoke2-base: 2 windows per micro-batch x 16. An artificial out-of-memory error at step 20, so the fallback is tested on the GPU; from there it should go on at 1 x 32.
  - LE-20261118-0128-smoke2-rank: 4 x 8, to see whether 4 fits now.
  - Both: evals at steps 0, 12, 24 and 25; HellaSwag on the first 500 items at the start and end; samples at the start and end; latest.pt every 10 steps; one float16 checkpoint and the final split checkpoint, all deleted after the run.
Hypothesis: the logits were the part of the memory that changed with the data; chunking takes about 2 GB off the peak at 4 per micro-batch.
Prediction:
  - Both runs exit 0.
  - Rank arm, 4 per micro-batch: peak 12.9 GB (12.0-13.8), and it fits all 25 steps (70%).
  - Base arm, 2 per micro-batch: peak 9.5 GB (8.5-10.5). One oom event at step 20, then 1 per micro-batch to the end.
  - Speed per T4: 2.95k tokens/s at 4 per micro-batch (2.7-3.2k), 2.85k at 2 (2.6-3.1k), 2.4k at 1 (2.0-2.8k).
  - Step 0: the same OASST and Dolly losses and pref_acc as in the first smoke test, within 0.001.
  - HellaSwag on the first 500 items at step 0: within 1.0 point of Level 3's 48.2% on the same items.
Rule for the main run's micro-batch (fixed now): 4 if the rank arm's peak stays below 13.5 GB for all 25 steps with no oom event; otherwise 2. The any-step fallback stays on either way.
Cost estimate: about 16 min of session, 0.27 GPU-h, $0. The week so far: 0.32 of 30 GPU-h.

## 2026-11-18 01:50 · EP · result: LE-20261118-0128-smoke2. The memory fixes work; each micro-batch carried 0.57 s of fixed cost, now found and removed
Actual: kernel ep04-chat-smoke2 version 1, pushed 01:27:52, started 10 s later, 972 s in all. Both runs exit 0.
  Rank arm, 4 per micro-batch: no out-of-memory error in 25 steps; peak 14.70 GB from step 2 on, the same as in the first smoke test; 2.93-3.03k tokens/s. Step 3, where the first smoke test crashed, now passes.
  Base arm, 2 per micro-batch: peak 11.21 GB, 2.11-2.13k tokens/s. The artificial out-of-memory error at step 20 was logged as an oom event, the step was redone at 1 per micro-batch, and the run went on to the end at 1.36-1.37k tokens/s.
  Step 0: OASST 2.2054, Dolly 2.8118, pref_acc 64.31%, all as in the first smoke test. The base arm's training losses at steps 1, 2, 3 and 10 (2.2990, 2.3704, 2.2704, 2.2295) equal the first smoke test's to 4 decimals: chunking the loss changed nothing in the numbers. HellaSwag on the first 500 items at step 0: 48.2% in both arms, exactly Level 3's 48.2% on the same items.
  After 25 steps (0.82M tokens): base OASST 1.9265, Dolly 2.3024, pref_acc 63.6%, HellaSwag-500 46.2%; rank OASST 1.9229, Dolly 2.3163, pref_acc 63.2%, HellaSwag-500 47.8%.
  What the step times say: 8 micro-batches per step took 10.9 s, 16 took 15.5 s, 32 took 24.0 s. That fits 0.57 s of fixed cost per micro-batch plus 6.4 s of compute per 32-window step (32 x 0.57 + 6.4 = 24.5). Without the fixed cost the T4 would do about 5.1k tokens/s.
  The cause, found on CPU: my port keeps the JAX layout, where each weight is stacked over the 24 layers, and each layer took its slice with self.wq[i]. In the backward pass autograd then builds a zero gradient the size of all 24 layers for every layer and every weight, about 100 GB of memory traffic per micro-batch. The fix takes the per-layer views once per forward pass (unbind) and passes them to the layers. Checked on CPU with the full-size model (32 tokens): forward plus backward 12.1 s before, 1.0 s after, the same loss and gradients (difference 0.0), generation unchanged.
Was I right?
  Both exit 0: yes.
  Rank arm's peak 12.9 GB (12.0-13.8): no, 14.70. The logits caused the spikes (the crash step now passes) but not the steady peak. Fits all 25 steps (70%): yes, with 0.9 GB to spare.
  Base arm's peak 9.5 GB (8.5-10.5): no, 11.21. One oom event at step 20, then 1 per micro-batch: yes.
  Speed: 4 per micro-batch 3.0k (2.7-3.2k): yes. 2 per micro-batch 2.12k (2.6-3.1k): no. 1 per micro-batch 1.37k (2.0-2.8k): no. I assumed the micro-batch size barely matters; here each one carried a fixed cost I had not looked for.
  Step 0 within 0.001: yes, identical. HellaSwag-500 within 1 point: yes, equal.
  The micro-batch rule fixed at 01:27 says 2 (peak 14.70 is not below 13.5). The code has changed since (the unbind), so a third smoke test comes first, with its own rule below.
What I'll change: the unbind (done, src/chat_model.py), then smoke test 3.
Cost: 0.29 GPU-h (972 s plus a 1-minute allowance), $0. This week: 0.61 of 30 GPU-h.

## 2026-11-18 01:50 · EP · LE-20261118-0151-smoke3: the smoke test once more, after the unbind fix
Decision: kernel ep04-chat-smoke3, 20 steps per arm: LE-20261118-0151-smoke3-base at 2 per micro-batch x 16, LE-20261118-0151-smoke3-rank at 4 x 8. Evals at steps 0, 10 and 20; HellaSwag on the first 500 items at the start and end; samples at the start and end; latest.pt every 10 steps; checkpoints deleted after.
Hypothesis: the fixed cost per micro-batch was the per-layer indexing of stacked weights. With it gone, speed depends little on the micro-batch size, and both get near the 5.1k tokens/s implied by smoke test 2.
Prediction:
  - Both exit 0, no oom events.
  - Speed: rank arm (4 per micro-batch) 4.8k tokens/s (4.2-5.5k); base arm (2) 4.5k (3.9-5.2k).
  - Peak memory: rank arm 14.0 GB (13.0-14.8); base arm 10.5 GB (9.5-11.3).
  - Step 0 as before; the base arm's training losses at steps 1, 2, 3 and 10 equal smoke test 2's within 0.0002 (the same sums, possibly added in another order).
Rule for the main run's micro-batch (fixed now): 2, unless 4 is more than 10% faster and its peak stays below 13.5 GB with no oom event.
Cost estimate: about 12 min of session, 0.22 GPU-h, $0. The week so far: 0.61 of 30 GPU-h.

## 2026-11-18 02:03 · EP · result: LE-20261118-0151-smoke3. The unbind fix works: 4.0k tokens/s at 2 per micro-batch, peak 10.4 GB
Actual: kernel ep04-chat-smoke3 version 1, pushed 01:49:57, started 4 s later, 591 s in all. Both runs exit 0, no oom events.
  Base arm, 2 per micro-batch: 4,007 tokens/s on average over steps 2-20 (3,901-4,068), peak 10.37 GB.
  Rank arm, 4 per micro-batch: 4,151 tokens/s (3,965-4,386), peak 14.02 GB.
  The base arm's training losses at steps 1, 2, 3 and 10 (2.2990, 2.3704, 2.2704, 2.2295) equal smoke tests 1 and 2 to 4 decimals. Step 0 as before (OASST 2.2054, Dolly 2.8118, pref_acc 64.31%, HellaSwag-500 48.2%).
  After 20 steps (0.66M tokens): base OASST 1.9386, Dolly 2.3155, pref_acc 62.9%, HellaSwag-500 46.6%; rank OASST 1.9343, Dolly 2.3291, pref_acc 62.8%, HellaSwag-500 46.8%.
Was I right?
  Exit 0, no oom: yes.
  Speed: 4 per micro-batch 4.15k vs 4.8k (4.2-5.5k), just below my range; 2 per micro-batch 4.01k vs 4.5k (3.9-5.2k), in range. The fixed cost is gone (the two sizes differ by 3.6%), but the T4 does not reach the 5.1k I extrapolated. Some of the compute in that estimate also scaled with micro-batches.
  Peak: 14.02 vs 14.0 GB (13.0-14.8), and 10.37 vs 10.5 GB (9.5-11.3). Right.
  Losses within 0.0002 of smoke test 2: yes, equal.
Rule fixed at 01:50: 4 per micro-batch only if more than 10% faster and peaking below 13.5 GB. It is 3.6% faster and peaks at 14.02 GB, so the main run uses 2 per micro-batch x 16.
Cost: 0.18 GPU-h (591 s plus a 1-minute allowance), /usr/bin/bash. This week: 0.79 of 30 GPU-h.

## 2026-11-18 02:03 · EP · LE-20261118-0201-base and LE-20261118-0201-rank: the main A/B fine-tune
Decision: kernel ep04-chat-main, both arms at once, one per T4, as in the plan of 00:49 (data, recipe, decision rule unchanged), with the code changes the smoke tests forced: the loss in chunks, the fallback at any step, the unbind, 2 windows per micro-batch x 16 (still 32 per step).
  LE-20261118-0201-base on GPU 0: 414 steps, 13.6M tokens. LE-20261118-0201-rank on GPU 1: 695 steps, 22.8M tokens.
  Evals about every 34 (base) and 57 (rank) steps; HellaSwag on all 10,042 items at the start, middle and end; samples at log-spaced steps; 3 float16 checkpoints; latest.pt every 25 min; final weights plus optimizer state. Each run stops itself cleanly after 270 min (--time_limit_min); the session limit is 5.5 h.
  Download: every file except the optimizer states (step_<n>_opt.pt stay on Kaggle).
Prediction. The plan's predictions of 00:49 stand and will be scored as written. These are updated after the smoke tests (step 0 and the first 25 steps are now known):
  - Base arm, final: OASST val loss 1.82 (1.72-1.92); Dolly 2.10 (1.95-2.25); pref_acc 62% (57-67); HellaSwag 39.6% (38.3-40.6).
  - Rank minus base: OASST val loss -0.03 (-0.07 to +0.01); pref_acc -1 point (-4 to +2); HellaSwag -0.3 (-1.0 to +0.5); Dolly +0.01 (-0.02 to +0.05). Rank adopted by the rule: 55%.
  - Length-matched pref_acc (pairs within 1.5x of each other's length, 403 pairs; secondary, not in the rule): base final 55% (48-62); rank minus base -1 point (-6 to +4).
  - Speed 4.0k tokens/s per T4 (3.8-4.2k); no oom event (90%).
  - Session 2.4 h (2.1-2.8): the rank arm's 1.6 h of training, plus evals, HellaSwag and samples.
Cost estimate: 2.4 GPU-h, /usr/bin/bash. This week after it: about 3.2 of 30 GPU-h.

## 2026-11-18 13:30 · EP · result: LE-20261118-0201-base and LE-20261118-0201-rank. The runners-up pass all four tests; the chat model is the rank arm (step 695)
Actual: kernel ep04-chat-main version 1, pushed 02:01:54, started 9 s later, 8,377 s in all (2.33 h). Both runs exit 0, no oom event, 2 per micro-batch throughout, peak 10.37 GB on each card.
  Base arm: 414 steps, 13.57M tokens, 5,519 s of training, 4,016 tokens/s (3,904-4,097).
  Rank arm: 695 steps, 22.77M tokens, 7,963 s of training, 3,999 tokens/s (3,456-4,123).
  Final eval (step 0 was OASST 2.2054, Dolly 2.8118, pref_acc 64.31%, HellaSwag 40.75%):
                       base (414)   rank (695)   rank - base   rule
    OASST val loss       1.8144       1.7895       -0.0249     (a) < -0.01: pass
    pref_acc             63.25%       62.67%       -0.58 pt    (b) >= -2 pt: pass
    HellaSwag acc_norm   42.26%       42.33%       +0.07 pt    (c) >= -1 pt: pass
    Dolly val loss       2.2092       2.2082       -0.0010     (d) <= +0.02: pass
  So the rank arm is the chat model (rule of 00:49): runs/LE-20261118-0201-rank/ckpt/step_695.pt.
  At equal tokens the rank arm is already ahead, so the gain is not only its extra steps. Base reached 1.8140 at 8.9M tokens and stayed there to the end (1.8144 at 13.6M, learning rate decayed to 0). Rank was at 1.7974 at 13.1M tokens, still at a high learning rate, and kept falling to 1.7895. A fourth and fifth pass over the base data would not obviously have done the same; I did not run that arm.
  HellaSwag, paired over the 10,042 items: rank - base +0.07 pt (SE 0.13; 93 items only rank gets right, 86 only base). Against the Level 3 model's own item file: base +1.57 pt (SE 0.26), rank +1.64 pt (SE 0.26). Most of the gain is on WikiHow items, +1.9 pt for both arms (39.11% -> 41.05 / 41.02); ActivityNet +0.8 / +1.1 pt (44.00% -> 44.80 / 45.08). WikiHow is how-to text, the kind of writing closest to Dolly and OASST answers.
  pref_acc fell for both arms from step 0 (64.3% -> 63.2 / 62.7). Length-matched (403 pairs within 1.5x): step 0 54.3%, base 55.6%, rank 55.8%, rank - base +0.25 pt (SE 0.56). Fine-tuning on the best replies did not make the model rank the volunteers' preferred replies higher; on all pairs its preference is mostly length.
  The rank arm's training loss (1.633 at the end, weighted) is not comparable with base's (1.756).
Was I right? Plan of 00:49:
  Base final OASST 2.20 (2.00-2.40): no, 1.814 (I had not yet seen step 0, which was already 2.21). Dolly 2.10 (1.85-2.40): yes, 2.209. pref_acc 55% (50-60): no, 63.2%. HellaSwag 39.7% (38.0-40.8): no, 42.3%: I expected forgetting and got a gain.
  Rank - base: OASST -0.02 (-0.06 to +0.02): yes, -0.025. pref_acc -1 (-4 to +2): yes, -0.6. HellaSwag -0.3 (-1.0 to +0.5): yes, +0.07. Dolly +0.01: -0.001, close. Rank adopted (55%): it was.
  Main run about 1.9 GPU-h: no, 2.34 (the T4 is slower than I planned for: 4.0k tokens/s, not 5.0k).
Updated predictions of 02:03:
  Base final OASST 1.82 (1.72-1.92): yes, 1.814. Dolly 2.10 (1.95-2.25): yes, 2.209. pref_acc 62% (57-67): yes, 63.2%. HellaSwag 39.6% (38.3-40.6): no, 42.3%.
  Rank - base: all four inside their ranges (OASST -0.025, pref -0.6, HellaSwag +0.07, Dolly -0.001).
  Length-matched: base 55% (48-62): yes, 55.6%. Rank - base -1 (-6 to +4): yes, +0.25.
  Speed 4.0k (3.8-4.2k): yes, 4.02k and 4.00k. No oom (90%): yes. Session 2.4 h (2.1-2.8): yes, 2.33 h.
What I learned:
  - The runners-up help a little and cost nothing in the other tests: -0.025 nats on held-out conversations, the size I guessed. The extra human replies are worth having even at a quarter or an eighth of the weight.
  - Supervised chat tuning made this model better at HellaSwag, not worse, mostly on WikiHow items. My "fine-tuning forgets" prior was wrong for data this close to the pretraining text.
  - pref_acc is not a useful test of this kind of tuning: it starts at 64% because of length and does not move.
Download problem: the first fetch at 04:24 lost its connection (DNS failure for www.kaggleusercontent.com) 1.12 GB into the rank arm's 1.37 GB final weights. The Kaggle CLI still exited 0, and tools/chat_job.py moved the partial file into runs/ as if it were whole; the rank arm's logs had not come down at all. I caught it at 13:20 from the file list and the size. Fix: fetch() now leaves any file with a .kaggle-partial marker in kaggle/out, merges only whole files, and exits 1. Second fetch at 13:24: the remaining 11 files and the kernel log and session.json. Both final checkpoints are 1,367,835,599 bytes and load.
Correction: in the two entries of 02:03 the cost lines read "/usr/bin/bash". They should read "$0". Same slip as on 2026-11-08: the shell expanded $0 in an unquoted heredoc. The entries stay as they are. This entry went through a quoted heredoc.
Files: runs/LE-20261118-0201-ab.json (rule, curves, paired HellaSwag, length-matched pref), runs/LE-20261118-0201-{base,rank}/.
Cost: 2.34 GPU-h (8,377 s plus a 1-minute allowance), $0. This week: 3.13 of 30 GPU-h; TPU 9.08 of 20 h.
Correction to the "Correction" line above: the earlier slip was noted on 2026-11-07 (entry of 22:37), not 2026-11-08.

## 2026-11-18 13:31 · EP · LE-20261118-1331-chat: I talk to the chat model
Decision: the chat model is runs/LE-20261118-0201-rank/ckpt/step_695.pt (float32 weights, the rule of 00:49). I talk to it through tools/chat.py on this PC's CPU with the same settings the untuned model had at 00:53: temperature 0.7, top-p 0.9, repetition penalty 1.1 over 64 tokens, at most 300 new tokens, reply k uses seed 1000 + k. Every turn is logged as typed and as generated (runs/LE-20261118-1331-chat/).
  Part 1, before/after: the seven questions of 00:53, each in a fresh conversation (q1.json .. q7.json), so each reply uses seed 1000, as the untuned model's did.
  Part 2, conversation_1.json: one conversation, me talking to it. Greeting; I tell it my name; questions that build on its earlier answers; I tell it how it was made (342M parameters, trained from scratch on FineWeb-Edu on Kaggle TPUs, then taught to chat on OpenAssistant and Dolly); I ask it to tell me back; later I ask what my name is. The window is 1,024 tokens; when the conversation outgrows it, the oldest exchanges drop out of its context (the file records how many).
Prediction:
  Part 1 (the untuned model got 0 of 7 on topic, 3 of 7 turn endings):
  - On topic in its first sentence: 6 of 7 (range 5-7). Ends its own turn: 7 of 7 (6-7).
  - Capital of France: Paris (95%). 7 x 8 = 56 (40%). Sky: mentions scattering of sunlight (50%).
  - "Who are you?": calls itself a chatbot, an assistant or an AI (75%); calls itself Open Assistant (15%); calls itself Claude (under 2%; no reply in its training data does). The plan's prediction "calls itself Open Assistant" is scored as written.
  - Poem: 4-16 lines about the sea, at least one rhyme (60%).
  Part 2:
  - Remembers my name when asked after 3+ turns, while it is still in the window: 55%.
  - Tells back at least two true facts of how it was made when I ask, right after I told it: 50%.
  - At least one reply in the conversation states a plain falsehood with confidence: 90%.
  - At least one reply loops (the same sentence 3+ times): 25%.
  - CPU speed 11 tokens/s (9-13); the untuned model ran at 11.7.
Cost: $0, CPU only, about 20 minutes.
(This entry was written at 13:29 by the clock; its header and run ID use 13:31, when the first question goes in.)

## 2026-11-18 13:40 · EP · result: LE-20261118-1331-chat. It chats: on topic 7 of 7 (untuned 0 of 7), ends its own turn 19 of 20; wrong at sums, and it took my name for its own
Actual: model runs/LE-20261118-0201-rank/ckpt/step_695.pt (sha256 c73eebd3bd17c4f2), CPU float32, settings as planned. The first question went in at 13:30:00, not 13:31 as the note under the plan says. 20 replies, 2,203 tokens in 210 s: 10.5 tokens/s.
  Part 1, the seven questions, each in a fresh conversation, seed 1000 (runs/LE-20261118-1331-chat/q1.json .. q7.json):
    q1 "Hi! Who are you?" -> "I am an AI language model, trained on vast amounts of data and text ..." Ended its turn, 34 tokens.
    q2 capital of France -> "The capital of France is Paris." Then made-up facts: a "Pierre Simonet, the first President of the Republic"; Paris "the fourth-most populous city in the world" with 7 million people.
    q3 sky -> on topic but wrong: refraction, diffraction, and the sky "reflects some of this light back from the earth". No scattering.
    q4 sleep -> five numbered tips (asked for three), all sensible.
    q5 poem -> nine lines about the sea, no rhyme.
    q6 7 x 8 -> "7 times 8 is 23."
    q7 AI research -> about AI research but does not answer; made-up history of "OpenAI, a project funded by the US Department of Energy"; ran to the 300-token cap.
    On topic in the first sentence: 7 of 7 (untuned: 0 of 7). Ended its own turn: 6 of 7 (untuned: 3 of 7). Answers the question asked: 5 of 7 (q6 wrongly, q7 not at all).
  Part 2, conversation_1.json, 13 exchanges, 13:32-13:36:
    - My first message said "My name is Claude." Its first reply: "My name is Claude, and I am an AI assistant ..." It took the name from my message.
    - I told it how it was made and asked it to tell me back. It said it is a language model trained on text "the Open Assistant community has created", mixed "you" and "I", and repeated "trained on the development of language in different contexts" three times. No 342M, no FineWeb-Edu, no Kaggle, no Databricks.
    - "What is 12 plus 5?" first got "I am not sure how you got that answer!", then "12 + 5 = 25".
    - Asked which topic it would pick, it listed eight (one twice) and did not pick. Asked to pick one, it did not.
    - Memory check: my name was in its window (the message where I said it again was 3 turns back). Reply: "I'm sorry, but I can't remember my name ... My name is Claude." The right word, attached to the wrong speaker.
    - Asked to ask me a question, it listed eight questions about "Claude", one of them about living in New York City.
    - "Can an AI do AI research?" -> "Yes, there are many different types of AI research projects", then a list of model families (deep learning, reinforcement learning, CNNs, ...).
    - My goodbye -> "I'm sorry, I don't understand your question."
    - It ended all 13 of its turns. By its 5th reply the first exchange had left the 1,024-token window; the last reply saw 8 of the 24 earlier turns.
Was I right?
  On topic 6 of 7 (5-7): 7, in range. Turn endings 7 of 7 (6-7): 6, in range, point wrong.
  Paris (95%): yes. 7 x 8 = 56 (40%): no, 23. Scattering (50%): no. "Who are you?" a chatbot, assistant or AI (75%): yes, "an AI language model". Open Assistant (15%): no. Claude in q1 (under 2%): no. In the conversation it did call itself Claude, but only after reading the name in my message. Poem with a rhyme (60%): no.
  Name recall while in the window (55%): no; it said the name but as its own.
  Two true facts of how it was made (50%): just yes. "Language model" and OpenAssistant are true; both were the vaguest things I told it.
  A confident falsehood (90%): yes, many. A loop (25%): yes, once (the same clause 3 times in one reply).
  Speed 11 tokens/s (9-13): yes, 10.5.
  The plan of 00:49: fluent and on topic for simple questions: yes. Ends its own turns: yes, 19 of 20. Wrong on facts beyond common knowledge: yes, and on sums a child can do. Calls itself Open Assistant: no.
What I learned:
  - Two and a half hours on a free T4 turned a web-text model into something that answers in the form of an answer, on the subject, and stops. What it still lacks is above form: facts (Paris yes, the rest invented), arithmetic, following a correction, and knowing who is speaking.
  - "Who" is the weakest part. In the OASST and Dolly data the user never says their name and the assistant never has one, so a name in the conversation is just a word to copy. A small model fine-tuned only on that data cannot tell "my name" from "your name" at all.
  - The 1,024-token window holds only about four exchanges of this length, with 300 tokens kept for the reply. A longer conversation needs shorter replies or a longer context; with 300 new tokens it forgets the start of the talk by the fifth reply.
  - These conversations are logs, not data: rule 9. Nothing written here, by me or by the model, goes into any training set.
Files: runs/LE-20261118-1331-chat/q1.json .. q7.json, conversation_1.json (both sides, timestamps, sampling settings and seed per reply, token counts, context drops; unedited).
Cost: $0, CPU only, about 6 minutes of generation. This week: 3.13 of 30 GPU-h; TPU 9.08 of 20 h.
(Corrections to this entry: it was written at 13:36 by the clock, not 13:40 as the header says; and the cost line should read "3.5 minutes of generation (210 s) in about 6 minutes of talking", not "about 6 minutes of generation".)
