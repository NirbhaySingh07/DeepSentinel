# DeepSentinel Quick Runbook (Do This In Order)

## 1) Create Hugging Face assets
```powershell
set HF_TOKEN=hf_xxx
python scripts/create_hf_assets.py --username yourname
```

## 2) Local sanity (already runnable)
```powershell
python -m tests.test_deepsentinel
```

## 3) Run a small training and capture log
```powershell
$env:PYTHONUTF8='1'
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python train.py --model sshleifer/tiny-gpt2 --episodes 24 --max_steps 2 --group_size 2 --output_dir ./deepsentinel_model_small 2>&1 | tee train.log
```

## 4) Generate plot images
```powershell
python scripts/plot_training_curves.py --log train.log --loss-out loss_plot.png --reward-out reward_plot.png
```

## 5) Upload model + artifacts
```powershell
set HF_TOKEN=hf_xxx
python scripts/upload_to_hf.py --model-repo yourname/deepsentinel-overseer-small
```

## 6) Deploy Space and verify `/health` + `/tasks`
```powershell
powershell -ExecutionPolicy Bypass -File scripts/deploy_space.ps1 `
  -SpaceRepoUrl "https://huggingface.co/spaces/yourname/deepsentinel-demo" `
  -SpaceUrl "https://yourname-deepsentinel-demo.hf.space"
```

## 7) Final submission text + links
Open and fill:
`artifacts/submission_pack.md`
