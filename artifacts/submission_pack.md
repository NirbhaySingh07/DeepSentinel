# DeepSentinel Hackathon Submission Pack

## Links (fill these)
- Space URL: `<https://huggingface.co/spaces/yourname/deepsentinel-demo>`
- Model URL: `<https://huggingface.co/yourname/deepsentinel-overseer-small>`
- Loss plot: `<https://huggingface.co/yourname/deepsentinel-overseer-small/blob/main/artifacts/loss_plot.png>`
- Reward plot: `<https://huggingface.co/yourname/deepsentinel-overseer-small/blob/main/artifacts/reward_plot.png>`

## Generated local artifacts
- `train.log`
- `loss_plot.png`
- `reward_plot.png`
- model folder: `deepsentinel_model_small/`

## Short baseline vs trained note (submission-ready)
DeepSentinel trains an overseer policy that judges four detector agents, including one adversarial agent that may flip labels with high confidence.  
The reward shaping balances correctness, consensus-use, adversarial detection, calibration, and explanation quality, so the policy is rewarded for careful oversight instead of blind majority-following.  
Baseline behavior relies mostly on heuristic consensus handling, while trained behavior is optimized against the full multi-component reward and can learn to flag adversarial disagreement patterns more reliably.  
This creates an interpretable oversight agent that is explicitly optimized for robust, disagreement-aware decision making.
