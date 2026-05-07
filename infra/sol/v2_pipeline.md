# V2 launch sequence

Once Modal OPRA repack finishes and crypto recorder has accumulated
~24h of data, run these steps in order. All commands run from the Sol
shell unless marked `[mac]`.

## 0. Pre-flight on Mac

```bash
[mac] cd /Users/nahom/market-pattern-bot/.claude/worktrees/unruffled-goodall-8f32f7
[mac] /Users/nahom/Library/Python/3.9/bin/modal app list  # confirm OPRA repack done
```

## 1. Pull OPRA real-ts shards from Modal → Mac → Sol

```bash
[mac] mkdir -p ~/sol_xfer/opra_v1_realts
[mac] for day in OPRA-20260420-CTDQCTDCGX OPRA-20260424-DLKDHYSC6M OPRA-20260424-KNQS6Y7C6N OPRA-20260424-MQMCQVT7UW OPRA-20260424-NHL6U5LFKY OPRA-20260424-VF7J9WEXGA; do
  python3 -m modal volume get tradefm-smoke-shards "databento_opra_v1_realts/$day" "$HOME/sol_xfer/opra_v1_realts/$day"
done

[mac] rsync -avh --progress -e "ssh -i ~/.ssh/sol_ed25519" \
  ~/sol_xfer/opra_v1_realts/ \
  nwoldege@login.sol.rc.asu.edu:/scratch/nwoldege/data/opra_v1_realts/
```

## 2. Pack accumulated crypto data → Sol

```bash
[mac] rsync -avh --progress -e "ssh -i ~/.ssh/sol_ed25519" \
  ~/sol_xfer/crypto_raw/ \
  nwoldege@login.sol.rc.asu.edu:/scratch/nwoldege/data/crypto_raw/

# On Sol:
export PATH=/scratch/$USER/miniconda3/envs/tradefm/bin:$PATH
cd /scratch/$USER/silicon-alpha
python -m odte.data.crypto_recorder pack \
  --raw-dir /scratch/$USER/data/crypto_raw \
  --out-dir /scratch/$USER/data/packed/crypto
```

## 3. Push factor equity → Sol

```bash
[mac] rsync -avh --progress -e "ssh -i ~/.ssh/sol_ed25519" \
  ~/sol_xfer/factor_equity/ \
  nwoldege@login.sol.rc.asu.edu:/scratch/nwoldege/data/packed/factor_equity/
```

## 4. Real-ts merge across all 5 modalities (on Sol)

```bash
python -m odte.data.multimodal_interleave \
  --inputs /scratch/$USER/data/opra_v1_realts/OPRA-20260420-CTDQCTDCGX,/scratch/$USER/data/opra_v1_realts/OPRA-20260424-DLKDHYSC6M,/scratch/$USER/data/opra_v1_realts/OPRA-20260424-KNQS6Y7C6N,/scratch/$USER/data/opra_v1_realts/OPRA-20260424-MQMCQVT7UW,/scratch/$USER/data/opra_v1_realts/OPRA-20260424-NHL6U5LFKY,/scratch/$USER/data/opra_v1_realts/OPRA-20260424-VF7J9WEXGA,/scratch/$USER/data/packed/es,/scratch/$USER/data/packed/spy_nbbo,/scratch/$USER/data/packed/spy_l3,/scratch/$USER/data/packed/crypto,/scratch/$USER/data/packed/factor_equity \
  --output /scratch/$USER/data/packed/multimodal_v2
```

(Modalities default to enumerate(0,1,...,N) — works since all sources
now have modality_id columns. No --fallback-modalities needed.)

## 5. Launch V2 training

```bash
sbatch /scratch/$USER/silicon-alpha/infra/sol/train_multimodal_v2.sh
squeue -u nwoldege
```

ETA: ~5.5h.

## 6. Eval V2

```bash
# Wait for training squeue empty, then:
python -m odte.data.extract_tokenizer_edges \
  --dbn-dir /scratch/$USER/data/ES/GLBX-20260505-3CHPNJTXX5 \
  --out-json /scratch/$USER/data/packed/es/tokenizer.json   # already done

# Run eval as a SLURM job (login node OOMs on 1.2B model load):
cat > /scratch/$USER/silicon-alpha/sol_eval_v2.sh <<'SH'
#!/bin/bash
#SBATCH --account=grp_jadriazo
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=sa-eval-v2
#SBATCH --output=/scratch/%u/sol_logs/eval_v2_%j.out
#SBATCH --error=/scratch/%u/sol_logs/eval_v2_%j.err

set -e
cd /scratch/$USER/silicon-alpha
export PATH=/scratch/$USER/miniconda3/envs/tradefm/bin:$PATH
export PYTHONUNBUFFERED=1

# Tighter cost model than V1 — drop bin-width slippage, use just spread.
python -m odte.eval.multimodal_eval \
  --ckpt /scratch/$USER/checkpoints/tradefm_524m_mm_v2/tradefm_524m_mm_v2 \
  --shards "/scratch/$USER/data/packed/multimodal_v2/shard_*.parquet" \
  --tokenizer-json /scratch/$USER/data/packed/es/tokenizer.json \
  --notional 1000 --cost-pct 0.01 --commission 1.30 --slippage-bins 0 \
  --max-windows 500
SH

sbatch /scratch/$USER/silicon-alpha/sol_eval_v2.sh
```

## Decision rule (from V1):

| AUC | meaning | action |
|---|---|---|
| < 0.60 | V2 also loses to LightGBM | something fundamental wrong; pivot |
| 0.60–0.64 | tied baseline | architecture not the bottleneck |
| 0.64–0.66 | beat baseline narrowly | publish V2 result, plan V3 |
| 0.66–0.70 | clear win | scale corpus, plan deployment infra |
| > 0.70 | suspicious | audit for look-ahead bias |
