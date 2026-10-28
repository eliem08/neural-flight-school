# Neural Flight School

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliem08/neural-flight-school/blob/main/colab.ipynb)

A hands-on, tiny-transformer lab for learning causal attention, tensor shapes, and visualization without downloading pretrained weights.

## Attention lab

    pip install -r requirements.txt
    python attention_lab.py --text "attention is a computation"

Exports one attention heatmap per layer plus outputs/attention-lab/report.json. Uses a seeded, randomly initialized model and byte-derived token IDs to keep the example self-contained.

    python attention_lab.py --text "hello transformer" --device cuda --seed 7

The visualizations are intentionally educational: because the model is untrained, they do not show learned linguistic behavior. Try changing width, heads, and layers in TinyAttentionModel and inspect the attention tensor shapes.

## Quickstart

    python main.py

## License

MIT.
