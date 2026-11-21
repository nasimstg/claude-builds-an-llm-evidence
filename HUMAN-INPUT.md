# EP04 human input ledger

Built on Nov 18 from the lab transcripts, because no producer log was kept. It lists every message the owner typed in the lab sessions. The source files are in `ep04/lab/transcripts/C--ep04-lab/`. Claude never used its ask-a-question tool in the lab, and no turn was interrupted, so these typed messages are the whole of the human input inside the lab.

## Before the lab (Nov 7, setup)

- **The owner gave:** a Kaggle API token and a Hugging Face write token. They also said the card wasn't working and Modal had $1 of credit, and approved JAX for Level 3 ("Yes, allow JAX").
- **A separate Claude Code session did the rest:** the producer-side session, not the lab. It built the Python environment, the Kaggle CLI and Stockfish 19, saved the tokens, and ran test jobs on Kaggle's T4s and TPU. It also wrote the brief and the logging spec from the owner's plan.

## Inside the lab: every typed message (Bangladesh time)

| Level | When | The owner typed | What it was |
|---|---|---|---|
| L1 | Nov 7 20:16 | the brief (4,120 characters) | the job |
| L1 | Nov 8 01:25 | "Did you use a base model or train a completely new model from scratch? why did you do give me a high level overview" (in a fork of the session) | a question. The fork was deleted on Nov 8, so only this line is known. In that fork the owner also gave a deadline of "a couple of days", which Claude saved to its memory. |
| L1 | Nov 8 01:42 | "Fail: train bigger" | verdict. Claude's own pass protocol gave the owner the final word. |
| L1 | Nov 8 13:29 | "Fail, move on" | verdict |
| L2 | Nov 8 14:13 | the brief | the job |
| L2 | Nov 8 19:00 | "check now" | status nudge |
| L2 | Nov 8 21:19 | "are you done?" | status nudge |
| L3 | Nov 10 20:19 | the brief | the job |
| L3 | Nov 13 01:22 | "kaggle run has been queued for 2 days now can you create a new one and run again" | nudge about the TPU queue |
| L3 | Nov 13 01:26 | "how do i pause the queue and run from the web myself" | a question. A "Fork of ep04-l3-s1" notebook appeared in the Kaggle account at 01:21, so the owner had opened it in Kaggle's web editor. |
| L3 | Nov 13 01:35 | "run again (by yourself)" | handed it back to Claude |
| L3 | Nov 15 10:55 | "check notebook statuses" | status nudge |
| L3 | Nov 15 11:00 | "Yes, push it" | approval Claude asked for |
| L3 | Nov 17 17:42 | "all the run should be finished now" | status nudge |
| L3 | Nov 17 17:42 | "." (in a fork) | stray |
| L3 | Nov 17 23:04 | "Yes, run the scan" | approval Claude asked for |
| Epilogue | Nov 18 00:21 | the brief | the job |
| Epilogue | Nov 18 01:29 | "Yes, delete them" | approval Claude asked for. C: had 5.5 GB free; Claude checked that 530 files in `kaggle/out` were byte-for-byte copies of `runs/` and asked to delete them (22 GB). This let it download both chat models' weights, not only the winner's. |
| Epilogue | Nov 18 01:31 | "ok, let me know when the smoke test finishes" | acknowledgement |
| Epilogue | Nov 18 13:20 | "Try again" | restart. The PC lost its internet at about 04:24 (DNS failures), and the session stopped at 04:48 on "Can't reach the API server". It sat idle for 8.5 hours until this message. |

## Counts

- **Lines of code written or edited by the owner:** 0, by the owner's own account, and nothing in the transcripts says otherwise.
- **Briefs pasted:** 4, one per session (Levels 1–3 and the Epilogue). Each is the same text with only the last line changed.
- **Other typed messages:** 16.
  - 5 status nudges (including the queue one)
  - 2 verdicts ("Fail")
  - 3 approvals ("Yes")
  - 2 questions
  - 1 hand-back
  - 1 restart after the internet outage ("Try again")
  - 1 acknowledgement
  - 1 stray "."
- **Actions outside the chat:** 2.
  - Opening the Level 3 notebook in Kaggle's web editor (Nov 13).
  - Talking to the finished chat model in a terminal (Nov 18, 13:53 onward), with the command Claude gave at the end of the Epilogue. Logged unedited as `runs/LE-20261118-1331-chat/manager_1.json`. This was the owner joining the experiment, not a fix.
- **Claude's own `interventions.jsonl`:** still 3 entries: the two Level 1 verdicts and the Kaggle web-editor attempt. It added none in the Epilogue ("No human interventions were needed"). It didn't count status nudges, approvals Claude itself asked for, the restart after the outage, or the setup tokens.

## For the on-screen counter

All three ways of counting are true, so the script should say which one it uses:
- **Claude's log:** 3
- **Human touches that changed what happened:** 8. These are the 2 verdicts, 3 approvals, the queue nudge, the web-editor attempt and the "Try again" restart.
- **Every typed message except the briefs:** 16

The full list is short enough to show on screen as receipts. The owner's summary is that they only gave a Kaggle key and a Hugging Face token, plus a few prompts when something failed or stopped. That matches this ledger. The one addition is that the owner made the two "Fail" calls on Level 1, because Claude's own rules handed that call to them.
