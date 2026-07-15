# Zenodo submission metadata (paste-ready)

CHOSEN APPROACH: direct upload, lean. TWO files (the PDF + the frozen source snapshot), core
metadata, pick the license, submit.

Files to upload:
  1. the compiled PDF (see Step 1)
  2. C:\Users\dongy\PrismFormer-paper-v1-src.zip  — the frozen source snapshot (tag paper-v1, 0.29 MB,
     all code + the paper + the small arithmetic data; the BabyLM corpus is excluded because no paper
     benchmark uses it). This is what makes the results reproducible independent of the live repo.

Step 1 (only thing not preppable here): compile `prismformer.tex` to PDF.
  Overleaf.com > New Project > Upload Project > drop in prismformer.tex > download the PDF.
Step 2: zenodo.org > New upload > drag in BOTH files > paste the fields below > Publish.

Also (so the paper's "tag paper-v1" reference resolves on GitHub): push the tag once you can:
  git push origin paper-v1

---

**Upload type:** Publication → Preprint

**Title:**
A Self-Replicating Swarm of Tiny AIs: Phasor-Face Transformers (PrismFormer) with Arithmetic as Algebra and Bit-Exact, Mergeable Gradients

**Authors / Creators:**
Chen, Dongyang Stephen — Affiliation: Evaluated Applications
(Add your ORCID if you have one; free at orcid.org, recommended — it permanently disambiguates you as the author.)

**Publication date:** the date you upload (Zenodo defaults to today).

**Description (paste the abstract):**
I present PrismFormer, a transformer-shaped sequence model in which every token, word or number, is a "phasor face": a bundle of unit complex numbers composed by one algebraic operation, per-component complex multiplication ("binding"). Numbers are encoded so that binding performs arithmetic: addition on a linear-phase band, multiplication on a log-phase band, with results read back by correlation, so a computed value is decodable rather than merely reachable. This numeric representation is a direct application of the frequency-domain (phasor) form of Holographic Reduced Representations (Plate 1995; 2003) and of Random Fourier Features (Rahimi & Recht 2007); my contribution is to make it the trainable substrate of a transformer whose dense maps are replaced by parameter-lean "relation-banks". Because that substrate can be evaluated read-only during backpropagation, gradients accumulate into a detached buffer: a minibatch split into shards and summed is bit-for-bit identical to the serial result, which enables a full-replication training swarm, in which every node holds the whole model on a plain CPU, that stays mathematically coherent across mismatched hardware, unlike sharded systems that need matched accelerators.

I report results against a parameter-matched dense transformer at each setting. Arithmetic emerges from the codec with no training (binding two number faces decodes to their exact sum and product). Given the same worked, column-by-column problem, PrismFormer learns multi-digit addition, subtraction, and multiply/divide by a digit and generalises to unseen operand pairs where a matched transformer stays near zero; its working is legible one column at a time with no trained probe. On comparison and relational tasks it generalises better than a transformer that fits the same training data (65.9% vs 49.1% held-out), and it is a competitive character language model at matched size. A two-sided ablation shows that freezing the numeric identity does not help a single operation in isolation but wins under shared multi-task load. All results are at approximately 100k parameters; out-of-range extrapolation and large-scale convergence of the full colony remain open. Every quantitative result is produced by a script in the repository and reproducible with the command given in its section.

**Keywords (targeted, 8):**
phasor; Vector Symbolic Architectures; Hyperdimensional Computing; Holographic Reduced Representations; transformer; neurosymbolic arithmetic; distributed training; PrismFormer

**Related identifier:**
- URL: https://github.com/EvaluatedApplications/PrismFormer  — relation: "is supplemented by this upload"

**License:**
Because the record now BUNDLES your source-available code (not just the paper text), set the record
license to your repo's own terms rather than a CC license (CC is not meant for software). In the Zenodo
license picker choose "Other (Not Open)" — or add a custom license — and note in the description that
the full terms are in the bundled LICENSE file. Citing a paper is permitted under any license, so this
stays fully citable while the code stays protected.
(If you would rather keep the paper text under a permissive CC license, that requires splitting into two
records — paper under CC BY-NC-ND, code snapshot under its own license — which is less lean; the single
protected record above is the recommended path.)
