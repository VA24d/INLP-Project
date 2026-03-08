# Running MUSE Benchmark on IIIT Ada HPC

These scripts have been prepared to run the Gemma 3 1B MUSE unlearning pipeline on the IIIT Ada HPC cluster using the SLURM scheduler.

## 1. Transfer files to Ada
From your local machine, use `scp` or `rsync` to transfer the `scripts` folder and the datasets to the Ada master node:
```bash
scp -r "scripts" vijay.s@ada.iiit.ac.in:~/inlp_project
```

## 2. SSH into Ada
```bash
ssh -X vijay.s@ada.iiit.ac.in
```

## 3. Prepare Environment
Once logged in, navigate to the folder:
```bash
cd ~/inlp_project/scripts
```
Edit the `run_muse.slurm` file to insert your Hugging Face Token:
```bash
nano run_muse.slurm
# Replace "your_hf_token_here" with your actual token
```

## 4. Submit the SLURM Job
Submit the batch script to the SLURM queue:
```bash
sbatch run_muse.slurm
```

## 5. Monitor Output
The script is configured with `flush=True` on print statements to write directly to the SLURM output file. You can monitor the live output stream with:
```bash
tail -f muse_unlearn.out
```

To check the overall status of your job in the SLURM queue:
```bash
squeue -u vijay.s
```
