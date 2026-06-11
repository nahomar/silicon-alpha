# Faculty-sponsor email template

Copy / edit and send. Subject + body below.

One email unblocks **both** ASU Sol (free A100/A30 on the cluster) and
NSF ACCESS Explore (Delta/Anvil/Expanse). Same letter-of-collaboration
satisfies both programs' grad-student sponsorship requirements.

---

## Subject (pick one)

- `Sponsorship request: ASU Sol HPC + NSF ACCESS for microstructure ML research`
- `Request for letter of collaboration: HPC access for options-market ML research`

---

## Body

> Dear Dr. [LAST NAME],
>
> I am Nahom Woldegebriel (ASURITE: nwoldege), an ASU [student level —
> e.g. "undergraduate in Computer Science" / "graduate student in
> Mathematics"]. I am writing to ask if you would be willing to serve as
> a faculty sponsor for my compute-access requests at ASU Research
> Computing (Sol cluster) and the NSF ACCESS Explore program.
>
> **What the research is.** I am training a decoder-only transformer
> (TradeFM, 524M parameters) on high-frequency options-market
> microstructure data — the goal is a foundation model for trade-flow
> dynamics that can be evaluated on next-tick prediction and on
> cross-venue arbitrage signals in prediction markets (Polymarket,
> Kalshi). The research is motivated by the HFT-industry observation
> (Hudson River Trading, Jane Street) that generative models of exchange
> message streams capture structural regularities that classical
> econometrics misses.
>
> **What I have already done.** The full pipeline is implemented and
> validated end-to-end on Modal (1× and 8× H100). Concretely:
>
> - 637 parquet shards of real OPRA cmbp-1 tape (636M rows / ~4.5B
>   tokens) tokenized with a streaming-quantile hybrid tokenizer.
> - A 40M-param baseline trained on this corpus reaches loss 1.97 in
>   2000 steps (below the uniform-random entropy floor of log(4096)=8.3
>   and below a bigram Markov synthetic floor of 2.48), indicating the
>   model learns real microstructure structure, not just drift.
> - Distributed training infrastructure (FSDP + NCCL + rank-partitioned
>   checkpointing) validated on 8× H100 with `world_size=8`.
>
> Code at [https://github.com/nahomar/silicon-alpha] (formerly
> silicon-alpha). Full Phase-2/3/4/5 design specs under `docs/`.
>
> **What I need.** The 524M pretraining run requires multi-node H100
> (or multi-GPU A100 equivalent) for a week of wall-clock training on
> ~100B tokens. Single-node consumer compute cannot reach this scale;
> commercial cloud quotes ~$38-50k on Google Cloud A3 Mega. ASU Sol
> provides this free of charge to ASU-affiliated researchers with
> faculty sponsorship, and NSF ACCESS Explore provides equivalent
> allocations on NCSA Delta / Purdue Anvil.
>
> **What I am asking from you.** A brief letter of collaboration
> confirming you are aware of the computational activity and willing
> to be listed as the faculty-of-record. The letter templates:
>
> - ASU Sol: https://links.asu.edu/getHPC (I will fill the form; you
>   would be named as sponsor)
> - NSF ACCESS Explore: standard PI letter on departmental letterhead
>   acknowledging my student status and the research scope (I can send
>   a one-paragraph draft for your review)
>
> If you have research interests adjacent to this (ML, HPC, quantitative
> finance, computational economics), I would be happy to discuss the
> work in more depth — even a 20-minute chat would be genuinely useful
> given the breadth of the project.
>
> Thank you for considering this.
>
> Best,
> Nahom Woldegebriel
> nwoldege@asu.edu
> [phone if comfortable]

---

## Who to send it to (prioritized)

Send to the most-likely-yes first; only move down the list if they
decline. Good candidates:

1. **Your most recent CS / Math / Stats professor** — you have a
   known relationship; they can say yes in 5 minutes.
2. **Any SCAI faculty working on ML systems or HPC** (ASU School of
   Computing & AI). Search faculty directory for "machine learning",
   "high-performance computing", "quantitative", "finance".
3. **Faculty advisor of any research group you've TA'd or interned for**.
4. **Department chair** if 1-3 don't respond — they can redirect.

## What to attach

- One-page PDF of this project's `README.md` (screenshot the Silicon
  Alpha diagram at the top).
- Link to the GitHub repo.

## What NOT to write

- Don't oversell ("I'm going to beat HRT"). Keep it factual:
  research project, methodology, compute needed.
- Don't minimize ("I'm just a student playing around"). You have
  real results already (the 40M loss 1.97 number).
- Don't commit them to responsibilities beyond the sponsor letter.
  The letter is low-cost; if they want more involvement, that's a
  bonus, not a precondition.
