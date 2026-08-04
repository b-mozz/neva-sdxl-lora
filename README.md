# Neva — SDXL LoRA for Painterly Game Art

A LoRA adapter that teaches Stable Diffusion XL the flat, wide-vista painterly style of
*Neva* / *Gris* (Nomada Studio) game art, plus a paired evaluation harness that measures
whether the style actually transferred.

Trained and evaluated on **Kaggle (free tier, dual T4) in late March 2026.**

This repository is documentation of a completed experiment. It contains the training
notebook, the evaluation notebook, the training set, and the measured results.

---

## Result in one line

The LoRA moves generations **28% closer** to the target style by Gram-matrix distance
(winning on 13 of 20 paired prompts) and changes the render substantially
(LPIPS 0.57), at a cost of **−0.032 CLIP** prompt adherence. The style transfer is
visually unambiguous; the metrics supporting it are directional, not statistically
strong at n=20. Full numbers and caveats in [Benchmark results](#benchmark-results).

---

## Training data

22 hand-captioned frames from *Neva* and *Gris*, each 3420×1924 (16:9), paired with a
`.txt` caption file. Captions are long-form and describe subject, palette, technique,
and mood — not just content — so the trigger token binds to the *style*, not to wolves
and red cloaks.

<table>
<tr>
<td width="50%"><img src="assets/training/neva_01.jpg" width="100%"></td>
<td width="50%"><img src="assets/training/neva_03.jpg" width="100%"></td>
</tr>
<tr>
<td><sub>panoramic mountain vista, a large bird soaring mid-flight with dark wings against sky, soft lavender and muted green palette, layered watercolor mountains…</sub></td>
<td><sub>contemplative moment, a woman in crimson cloak offering flowers to a white wolf companion with flying petals, soft pink and coral gradient sky…</sub></td>
</tr>
<tr>
<td><img src="assets/training/neva_07.jpg" width="100%"></td>
<td><img src="assets/training/neva_08.jpg" width="100%"></td>
</tr>
<tr>
<td><sub>mystical shrine interior, a small wolf-like creature atop a glowing white monument in vertical space, deep midnight blue with cyan light beams…</sub></td>
<td><sub>open field landscape, a lone figure in red cloak running through vibrant yellow flower field toward distant purple mountain, golden yellows with lavender peaks…</sub></td>
</tr>
<tr>
<td colspan="2"><img src="assets/training/neva_11.jpg" width="100%"></td>
</tr>
<tr>
<td colspan="2"><sub>underwater cavern, a white wolf visible on upper ledge within cyan-lit cave space, monochromatic deep blue with concentrated turquoise glow, smooth gradient washes defining cave architecture…</sub></td>
</tr>
</table>

<sub>5 of 22. Captions abridged; each is prefixed with `a TOK style` during training.
Full set in [`training_dataset/`](training_dataset/).</sub>

The set is deliberately narrow in subject (landscapes, a small figure, a white animal)
and deliberately wide in palette — cool blues, warm ambers, pinks, monochrome — so the
adapter learns *how* the art is rendered rather than *what* it depicts.

---

## Method

### Training — [`training-neva.ipynb`](training-neva.ipynb)

Pivotal tuning via the `diffusers` advanced DreamBooth LoRA script: a low-rank adapter
on the UNet plus two learned textual-inversion tokens (`<s0><s1>`) in both text encoders.

| | |
|---|---|
| Base model | `stabilityai/stable-diffusion-xl-base-1.0` |
| VAE | `madebyollin/sdxl-vae-fp16-fix` |
| Script | `train_dreambooth_lora_sdxl_advanced.py` (diffusers, main) |
| LoRA rank | 8 |
| Optimizer | Prodigy, lr 1.0 (UNet and text encoder) |
| Text-encoder TI | enabled, `train_text_encoder_ti_frac=0.5` |
| Resolution | 1024, batch size 1, grad accum 4, repeats 5 |
| SNR gamma | 5.0 |
| Precision | fp16, gradient checkpointing, xformers |
| Steps | 2000 configured, checkpoints every 500 |
| Seed | 42 |

Evaluation used the **1000-step checkpoint** (`lora-neva-1000`).

Learned tokens are exported as an embeddings `.safetensors` and loaded at inference via
`load_textual_inversion` into both `text_encoder` and `text_encoder_2`.

### Evaluation — [`testing-neva.ipynb`](testing-neva.ipynb)

A **paired** design: the same 20 prompts, the same seed per prompt index (`42 + i`), the
same sampler settings, run once through vanilla SDXL and once through the LoRA. Only the
style phrase in the prompt differs:

- vanilla → `a painterly {prompt}`
- LoRA → `a <s0><s1> style {prompt}`

Prompts are drawn 2-per-category from a 500-prompt bank spanning 10 content categories
(natural environments, urban, characters, animals, still life, weather, fantasy,
interiors, abstract, seasonal), so a 20-image run still covers the full content space
instead of clustering in one domain.

| | |
|---|---|
| Prompts | 20 (2 × 10 categories) |
| Images | 20 vanilla + 20 LoRA, 1024×1024 |
| Steps / CFG | 30 / 7.0 |
| Base seed | 42, paired across arms |
| Negative prompt | photorealistic, 3D render, sharp edges, cartoon, anime, low quality, blurry, text, watermark, UI elements |

Four metrics, of which two are load-bearing at this sample size:

1. **Gram-matrix style loss** *(primary)* — VGG19 Gram matrices at conv1_1…conv5_1,
   compared against the mean Gram of the 22 training images. Each layer is normalized by
   its own reference Gram energy, otherwise the largest-magnitude layer decides the score
   alone. Lower = closer to the target style.
2. **LPIPS** *(paired)* — perceptual distance between the vanilla and LoRA render of the
   same prompt and seed. Measures how much the adapter changed the image; says nothing
   about direction.
3. **CLIP score** — cosine similarity to the **content-only** prompt. Both arms are
   scored against the same text, so this measures prompt adherence rather than the style
   wording. (Scoring the LoRA arm against its own prompt would feed the untrained
   `<s0><s1>` tokens into CLIP's text encoder and depress the score for reasons unrelated
   to the image.)
4. **FID** — computed only at n ≥ 200. **Skipped in this run**; at n=20 FID is dominated
   by sample-size bias and would report a function of the sample count, not model quality.

---

## Vanilla SDXL vs. Neva LoRA

Same prompt, same seed, same steps. Left is vanilla SDXL, right is the LoRA.

| Prompt | Vanilla SDXL | Neva LoRA |
|---|---|---|
| misty forest at dawn with sunlight filtering through tall pines | <img src="assets/comparisons/0000_vanilla.jpg" width="330"> | <img src="assets/comparisons/0000_lora.jpg" width="330"> |
| pine forest blanketed in fresh snow at night | <img src="assets/comparisons/0001_vanilla.jpg" width="330"> | <img src="assets/comparisons/0001_lora.jpg" width="330"> |
| a solitary figure walking along a beach at sunset | <img src="assets/comparisons/0004_vanilla.jpg" width="330"> | <img src="assets/comparisons/0004_lora.jpg" width="330"> |
| a white fox sitting in a snowy landscape | <img src="assets/comparisons/0006_vanilla.jpg" width="330"> | <img src="assets/comparisons/0006_lora.jpg" width="330"> |
| a stag with large antlers in morning mist | <img src="assets/comparisons/0007_vanilla.jpg" width="330"> | <img src="assets/comparisons/0007_lora.jpg" width="330"> |
| fresh snow on rooftops in a quiet town | <img src="assets/comparisons/0011_vanilla.jpg" width="330"> | <img src="assets/comparisons/0011_lora.jpg" width="330"> |
| a floating island with a waterfall pouring into the void | <img src="assets/comparisons/0012_vanilla.jpg" width="330"> | <img src="assets/comparisons/0012_lora.jpg" width="330"> |
| a city built on the back of a colossal turtle | <img src="assets/comparisons/0013_vanilla.jpg" width="330"> | <img src="assets/comparisons/0013_lora.jpg" width="330"> |
| ice crystal formations under a microscope | <img src="assets/comparisons/0017_vanilla.jpg" width="330"> | <img src="assets/comparisons/0017_lora.jpg" width="330"> |
| a scorching summer desert under a white sun | <img src="assets/comparisons/0019_vanilla.jpg" width="330"> | <img src="assets/comparisons/0019_lora.jpg" width="330"> |

**What transferred.** Flat color fields replacing photographic texture; a compressed,
desaturated palette; hard silhouettes against soft atmospheric washes; and — most
distinctly — the *compositional* habit of the source material: the subject shrinks and
the environment takes over. The fox, the stag, and the beach figure all become small
elements in a wide scene, which is exactly how the training frames are staged. That is a
composition prior being learned from 22 images, not just a texture filter.

**What broke.** The last two rows are the failure mode. "A city built on the back of a
colossal turtle" loses the turtle to a busy line-drawn cityscape, and "ice crystal
formations under a microscope" flattens into gray lichen shapes. Where a prompt needs a
single dominant foreground subject, the learned "small subject, vast environment" prior
fights it — and the environment wins. This is the mechanism behind the CLIP drop below;
it is a real regression, not measurement noise.

---

## Benchmark results

n = 20 paired prompts. Raw output: [`results/eval_outputs/eval_results.json`](results/eval_outputs/eval_results.json).

| Metric | Vanilla | LoRA | Δ | Direction |
|---|---|---|---|---|
| **Gram style loss** vs. training set | 4.436 ± 7.192 | **3.196** ± 3.340 | **−27.96%** | lower is better ✅ |
| — paired win rate | — | **13 / 20 (65%)** | — | higher is better |
| **CLIP score** (content-only prompt) | **0.3561** ± 0.0350 | 0.3238 ± 0.0358 | **−0.0323** | higher is better ❌ |
| **LPIPS** (vanilla ↔ LoRA) | — | 0.5688 ± 0.0978 | — | magnitude of change |
| **FID** | — | — | — | skipped (n=20 < 200) |

### Reading these honestly

**The style improvement is real but the mean overstates it.** A 28% reduction in style
loss sounds decisive, but vanilla's standard deviation (7.19) is larger than its mean
(4.44) — the average is being dragged by a few images that are wildly far from the target
style, and the LoRA fixing those inflates the percentage. The paired win rate is the more
honest statistic: **13 of 20**. Under a sign test that is p ≈ 0.26 — consistent with a
genuine effect, but not on its own evidence of one. What makes the effect credible is not
the p-value; it is that the side-by-side images are unmistakable and the LoRA's variance
collapses from 7.19 to 3.34, meaning it produces a far more *consistent* style than
vanilla does.

**The CLIP regression is the honest cost.** −0.032 is roughly one standard deviation of
the per-image spread, and it is not evenly distributed. The categories that lose most are
`characters_figures` (−0.071), `fantasy_surreal` (−0.086), and `abstract_compositional`
(−0.043); `interior_scenes` (+0.007) and `weather_atmospheric` (+0.017) actually improve.
That pattern lines up exactly with the visual failure mode: prompts needing a prominent
foreground subject suffer, atmospheric and environmental prompts do not. Per-category
numbers are 2 prompts each, so treat the pattern as a hypothesis to test at larger n, not
a finding.

**LPIPS 0.57 is a sanity check, not a win.** It confirms the adapter substantially
changes the image rather than nudging it. A high LPIPS with a *worse* style loss would
have meant the LoRA was breaking the image; here it moves with the style improvement, so
it corroborates.

**FID was deliberately not reported.** At 20 images per set, FID's bias term dominates
its signal, and the resulting number would mostly encode the sample size. The harness
skips it below 200 and records why. Reporting it would have been the easiest way to make
this look more rigorous than it is.

### What this run is and is not

This is a **smoke test that the training worked**, not a benchmark of how well it worked.
20 paired prompts is enough to confirm the adapter loads, the tokens resolve, the style
transfers, and to surface a failure mode — all of which it did. It is not enough to put a
confidence interval on the effect size, and no amount of metric selection makes it so.

To make the numbers publishable rather than indicative:

- Raise `PER_CATEGORY` to 20 in the prompts cell (200 images/arm) — enough for FID and for
  per-category CLIP to mean something.
- Add a human or VLM side-by-side preference test on the same pairs. None of these four
  metrics actually measures "does this look like *Neva*"; Gram loss measures texture and
  color statistics, which correlate with style but reward a model that merely matches the
  palette. A preference test measures the thing itself and is cheap at this scale.
- Sweep LoRA scale (0.4 / 0.6 / 0.8 / 1.0) to find where style gain stops being worth the
  prompt-adherence loss. The CLIP regression above is a single point on a curve nobody has
  plotted yet.
- Evaluate at 16:9 rather than 1:1. Training frames are all 3420×1924, generation is
  square, and the Gram comparison resizes both to 256×256 — so aspect-ratio distortion is
  folded into the style metric for every image.

---

## Repository layout

```
├── training-neva.ipynb           # LoRA + pivotal tuning training run
├── testing-neva.ipynb            # prompt bank, generation, evaluation
├── training_dataset/             # 22 PNG frames + .txt captions
├── assets/                       # downscaled images used in this README
└── results/
    └── eval_outputs/
        ├── eval_results.json     # all metrics from the run
        ├── generation_meta.json  # prompts, seeds, sampler settings
        ├── vanilla/0000-0019.png # 1024×1024 baseline renders
        └── lora/0000-0019.png    # 1024×1024 LoRA renders
```

## Reproducing

Both notebooks are written for Kaggle and reference `/kaggle/input/...` paths; adjust
`BASE_MODEL`, `LORA_PATH`, `EMBED_PATH`, and `TRAINING_DATA_DIR` for another environment.

```
pip install clean-fid lpips open_clip_torch torchvision diffusers transformers
```

`testing-neva.ipynb` runs top to bottom: cell 1 writes `prompts.py`, cell 2 generates both
image sets (resumable — it skips files already on disk), cell 3 evaluates. The evaluation
cell runs both as a notebook cell and as `python evaluate.py --skip-fid`.
