# Neural Flight School

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eliem08/neural-flight-school/blob/main/colab.ipynb)

A hands-on, tiny-transformer lab for learning causal attention, tensor shapes, and visualization without downloading pretrained weights.

## Interactive dashboard

    pip install -r requirements.txt
    streamlit run dashboard.py

Enter a toy sequence, generate per-layer attention heatmaps, and inspect tensor shapes and entropy summaries.

## Attention lab

    python attention_lab.py --text "attention is a computation" --device auto --seed 7

Exports heatmaps and a JSON report. The model is randomly initialized and untrained, so its visualizations demonstrate tensor mechanics only and do not show learned linguistic behavior.

## License

MIT.
