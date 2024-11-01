# MT_HW5

### Motivation
We implement the seq2seq model with attention mechanism using the Transformer model provided by PyTorch.

### Training Details

The neural machine translation model was trained on a CPU for a total of 150 epochs, achieving a final training loss of 0.015. During the training process, two decoding strategies were employed: greedy search and beam search.

Greedy Search

	•	Batch Size 32: Training took approximately 56.05 seconds per epoch.
	•	Batch Size 64: The time per epoch was reduced to 41.80 seconds.

Beam Search

	•	Batch Size 32: Each epoch required around 55.2 seconds.
	•	Batch Size 64: The training time improved to 42.1 seconds per epoch.

The results of the training were evaluated using the `BLEU` score, where the model achieved a score of `0.5314`. The final output is in the `translations` file.

### Usage

Run the following code in a terminal:
```
python seq2seq.py
```


### Team Members:

Anushri Suresh - asures13@jh.edu
Suhas Sasetty - ssasett1@jh.edu
Yogeeshwar Selvaraj - yselvar1@jh.edu