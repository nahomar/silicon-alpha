# Running on ASU Sol

Sol is ASU Research Computing's cluster (A100 GPUs, free for ASU users).
None of the current 0DTE models *need* it: ridge, LightGBM and the ~30k-param
sequence transformer run on a Mac. It matters for heavy jobs: retraining
TradeFM (the plan priced cloud GPUs at $38-50k; Sol is free), fine-tuning a
language model on millions of posts, or large hyper-parameter sweeps.

## Account (checked 2026-11-08)

- Personal settings (ASURITE id, Slurm account, key path) go in
  `infra/asu/sol.env`, which is gitignored; see `sol.env.example`.
- Login `<asurite>@login.sol.rc.asu.edu` with an SSH key; a Duo push arrives on each new login.
- Use the research-group Slurm account (QOS debug / private / public) that `check_sol.sh` lists.
- Home: 100 GB. System python is 3.6, so use the `odte` mamba env.
- One Duo approval can cover many commands: add
  `-o ControlMaster=auto -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=4h`.

## Connect

1. **VPN first.** `login.sol.rc.asu.edu` has no public DNS. Open Cisco Secure
   Client -> `sslvpn.asu.edu` -> ASURITE + Duo. (`sol.asu.edu` is a public
   web page, not the SSH host.)
2. Check: `infra/asu/check_sol.sh` (read-only). It confirms DNS, key login,
   and prints partitions, GPU types, your QOS and storage.
3. Optional `~/.ssh/config` entry:

       Host sol
         HostName login.sol.rc.asu.edu
         User <asurite>
         IdentityFile ~/.ssh/<your sol key>

## One-time environment on Sol

    module load mamba/latest
    mamba create -n odte python=3.11 -y && source activate odte
    pip install numpy pandas scipy scikit-learn lightgbm pyarrow torch transformers yfinance

## Code and data

Code: `git clone https://github.com/nahomar/silicon-alpha` on Sol.
Data is not in git (licensed quotes, social text). Copy it:

    rsync -av data/odte_0dte data/calendar data/context data/social/agg \
          data/social/nlp data/alt sol:~/silicon-alpha/data/

Then `sbatch -A <your account> infra/asu/sol_job.sbatch`. `odte/eval/seq_model.py` picks
CUDA automatically when a GPU is present.
