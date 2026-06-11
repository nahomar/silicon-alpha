# GCP A3 Mega quota request — fill-in template

Paste into the Google Cloud support-case form at
https://console.cloud.google.com/support/cases/create, or into the
"Request Adjustment" dialog on
https://console.cloud.google.com/iam-admin/quotas.

Fill the bracketed fields yourself. The free-form justification below
is calibrated to what GCP actually approves — vague requests ("for ML
training") get bounced; requests with a concrete model, step count,
and spend commitment clear quickly.

---

## Form fields

- **Service:** Compute Engine API
- **Quota metric:** `compute.googleapis.com/nvidia_h100_mega_gpus`
  *(alt: `nvidia_h100_mega_80gb_gpus` — name drifts, pick the one that
  matches "NVIDIA H100 80GB Mega" in the filter)*
- **Region:** `us-central1`  *(or `us-east5` — whichever region shows
  A3 Mega availability in your project's quota page today)*
- **Current limit:** [whatever the quota page shows]
- **Requested limit:** **8 GPUs** (one a3-megagpu-8g instance)
- **Dimension:** per region
- **Reason for increase:** see justification below
- **Expected duration:** 2 weeks
- **Contact:** [your email]

## Justification text (paste this verbatim)

> We are running a one-shot pretraining job for a 524M-parameter
> decoder-only transformer on options-market microstructure data. The
> workload is a single a3-megagpu-8g instance (8× H100 80GB SXM) with
> FSDP sharding, bf16 mixed precision with optional FP8 via
> transformer-engine, FlashAttention-3, and activation checkpointing.
>
> Scope:
>   - One pretraining run of approximately 200,000 optimizer steps
>   - Context length 4096, total batch size 16, gradient accumulation 4
>   - Training corpus ~20B tokens of CBOE OPRA options tick data,
>     delivered via Databento
>   - Expected wall time 10-14 days on the 8-GPU node
>   - Total expected compute spend: ~$16,000 (at on-demand rates)
>
> We specifically need the A3 Mega SKU (as opposed to A3 Standard or
> H100 High-GPU) because GPUDirect-TCPX is required to keep the FSDP
> all-gather latency under ~1.5 seconds per step; without TCPX the
> all-reduce bottleneck extends the wall time into the 20+ day range,
> substantially increasing both compute spend and schedule risk.
>
> The pretrain scaffolding has already been validated end-to-end on a
> smaller 40M-parameter model on a Colab A100 (5,000 steps, monotone
> loss curve, clean shard-checkpoint round-trip). Code is open in
> https://github.com/nahomar/silicon-alpha — specifically
> odte/train/distributed.py for the FSDP launcher and
> configs/tradefm_524m.yml for the architecture spec.
>
> Billing is set up on the project and we have adequate budget for
> the full 2-week burst. We do not need sustained quota beyond this
> single run — please approve for a 2-week window, and we will release
> the instances immediately on completion.

## Follow-up if the first answer is "need more information"

GCP Quota support sometimes asks for:

1. **Business justification.** Answer:
   > This is research-stage engineering for a single-operator
   > algorithmic-trading project. The 524M pretrain is the one-time
   > compute-heavy step that unblocks downstream live paper trading
   > on a separate, much smaller cluster.

2. **Why not cheaper GPUs?** Answer:
   > 524M FSDP at ctx_len=4096 requires ≥80 GB HBM per rank during
   > activation-checkpointed forward+backward. H100 80GB is the
   > lowest-spec GPU that fits. A100 80GB would also technically fit
   > but TCPX (required for our all-reduce budget) is an H100-only
   > network feature on GCP.

3. **Are you willing to commit to reservations / CUDs?** Answer:
   > Not at this stage. If the Phase-2 pretrain produces a model we
   > want to iterate on, we would consider CUDs at Phase-4.

## Expected turnaround

- Simple quota increase within a project's existing commitment: usually
  <24 h.
- New-GPU-family increase (A3 Mega often counts as new): 1–3 business
  days. Sometimes a same-day approval if there's capacity in the
  region.
- If region is capacity-constrained, you'll be offered an alternate
  region — us-east5 or us-central1 usually have the most A3 Mega
  availability.

## Track record / warning signs

- If the response says "please provide more information about your
  use case" — paste the follow-up section above. That usually clears it.
- If the response says "we are unable to approve A3 Mega at this time
  due to capacity" — pivot to RunPod 8×H100 as Gate-3 fallback, and
  resubmit the GCP quota for the following week. RunPod's TCP-only
  NCCL is ~25–35% slower but works.
