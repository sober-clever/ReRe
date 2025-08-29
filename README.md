# Implementations of  Reinforced Preference Optimization for Recommendation (ReRe).

## File Description 

- sft.py: the SFT code
- rere.py: the ReRe code
- rere_trainer.py: the grpo trainer tailored for recommendation

The training instructions can be seen in rere.sh and train.sh, while the evaluation instructions are in evaluate.sh.

## Quickstart

- Create a vairtual python environment.

```bash
conda create -n ReRe
```

- Install required packages.

```bash
pip install -r requirements.txt
```

- Execute the ReRe the training bash.

```bash
bash rere.sh
```

- Run the evaluation bash.

```bash
bash evaluation.sh	
```

