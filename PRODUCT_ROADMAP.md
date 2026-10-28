# Neural Flight School — Product Roadmap

## Product outcome

An interactive educational app that teaches transformer internals through runnable, visual lessons and progressively more realistic experiments.

## Users and problem

- Students learning deep learning and attention.
- Developers who want to understand tensor shapes and model blocks.
- Instructors needing small reproducible labs that run on free or local compute.

## Current foundation

- Tiny randomly initialized causal attention model, heatmaps, entropy summaries, Streamlit dashboard, Colab notebook.

## Roadmap

### Phase 1 — Learning experience
- [ ] Add guided lessons with learning objectives and expected tensor shapes.
- [ ] Add sliders for width, heads, layers, sequence length and seed.
- [ ] Add attention mask demonstrations and unit tests for causal masking.
- [ ] Add short quizzes and explanation panels after each lesson.
- [ ] Add a downloadable lab report.

### Phase 2 — Bridge to real models
- [ ] Add optional pretrained model mode, separate from the toy model.
- [ ] Compare untrained and trained attention maps with clear caveats.
- [ ] Add embedding, residual stream, MLP and logits visualizations.
- [ ] Add exercises with expected outputs and automated checks.
- [ ] Add accessibility, mobile layout and notebook lesson sequence.

### Phase 3 — Final product
A free interactive transformer course where learners can change architecture parameters, see live tensor shapes and attention maps, complete exercises, and export reproducible notebooks.

## Success criteria

- Core lessons run on CPU with no model download.
- Every visualization explains what it shows and what it cannot prove.
- Students can finish a guided lesson in under 15 minutes.
