# Plan: Re-train Laya Model for WCAG 2.2 Violation Detection

**Repository:** https://github.com/turbolego/layaForWeb
**Goal:** Fine-tune the Laya model to classify web accessibility violations per WCAG 2.2 criteria, deploy via Kaggle, and publish the resulting model to the layaForWeb repository.

---

## Phase 1: Create Kaggle Notebook

### 1.1 Setup
- Create new Kaggle notebook: `laya-wcag-training`
- Configure compute: GPU T4 x2 (Kaggle provides free)
- Install dependencies:
  - `torch` (PyTorch)
  - `transformers` (HuggingFace)
  - `datasets` (HuggingFace)
  - `onnx` (for model export)
  - `onnxruntime` (for verification)

### 1.2 Data Preparation
- Source datasets for training:
  - **Option A:** Use WCAG violation corpus (existing labeled datasets from WAI-ARIA, axe-core datasets)
  - **Option B:** Generate synthetic training data using LLM-augmented labeling
  - **Option C:** Combine existing accessibility evaluation datasets
- Create training data in format: `{text: str, violations: list[WCAG_criteria]}`
- Fields per violation:
  - `criterion` (e.g., "1.1.1", "1.2.1")
  - `principle` (Perceivable, Operable, Understandable, Robust)
  - `level` (A, AA, AAA)
  - `description` (natural language violation description)

### 1.3 Next Steps
- [x] Create initial Kaggle notebook with data loading skeleton
- [x] Add data loading for WCAG 2.2 training corpus
- [x] Add model architecture (multi-label classification heads)
- [x] Add training loop
- [x] Add evaluation metrics
- [x] Add ONNX export and quantization

---

## Phase 2: Model Training

### 2.1 Architecture Adaptation
- Load base Laya model (ModernBERT-large encoder + 2-layer transformer head)
- Adapt for multi-label classification:
  - Add classification heads for each WCAG criterion
  - Output: probabilities for each WCAG 2.2 criterion (100+ criteria)
  - Yes/No decision per criterion
  - Confidence score per prediction

### 2.2 Training Pipeline
```python
# Training loop structure
for epoch in range(NUM_EPOCHS):
    # Forward pass with training data
    outputs = model(input_text, labels=violation_labels)
    
    # Loss calculation
    loss = binary_cross_entropy_with_logits(outputs, target_labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Evaluation metrics
    precision, recall, f1 = evaluate(model, validation_data)
```

### 2.3 Hyperparameters
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3-5
- Optimizer: AdamW
- Weight decay: 0.01
- Scheduler: Linear warmup + decay

---

## Phase 3: Model Evaluation

### 3.1 Metrics
- **Primary:** F1-score (macro-averaged across all WCAG criteria)
- **Secondary:**
  - Precision per criterion level (A, AA, AAA)
  - Recall per criterion level
  - False positive rate
  - False negative rate

### 3.2 Validation Sets
- Hold-out test set: 20% of training data
- Cross-validation: 5-fold
- External validation: axe-core test cases, WAI website examples

### 3.3 Error Analysis
- Identify criteria with low recall
- Check for systematic false positives
- Analyze confusion matrix for common violations

---

## Phase 4: Model Export & Optimization

### 4.1 ONNX Export
```python
import torch
import onnx

# Export trained model
torch.onnx.export(
    model,
    dummy_input,
    "laya-wcag.onnx",
    opset_version=17,
    input_names=['input_ids', 'attention_mask'],
    output_names=['criterion_scores', 'confidence']
)
```

### 4.2 Quantization
- Apply weight-only quantization (similar to q8e8 build in layaForWeb)
- Verify accuracy retention (>95% of float32 performance)
- Target model size: <400MB

### 4.3 Manifest Generation
- Split into 24 MiB parts
- Generate SHA-256 hashes
- Create `manifest.json` for browser caching

---

## Phase 5: Integration with layaForWeb

### 5.1 Repository Updates
- Add new notebook to `/notebooks/`
- Add trained model weights to releases or HuggingFace Hub
- Update `serve.py` to load WCAG-specific model
- Add configuration for WCAG violation detection mode

### 5.2 API Extension
- Extend `systemOne()` to handle WCAG detection:
```javascript
const questions = {
  wcag_violations: {
    type: "choice",
    instructions: "Which WCAG 2.2 criteria are violated?",
    criteria: wcag_criteria_list  // 100+ criteria
  }
};
```

### 5.3 Documentation
- Add `docs/md/laya-wcag-training.md` explaining:
  - Training process
  - Model capabilities
  - Usage examples
  - Limitations
- Update README with WCAG detection as new use case

---

## Phase 6: Deployment & Publishing

### 6.1 Kaggle Publishing
- Push notebook to Kaggle
- Set as public (or private with share link)
- Document model weights download link

### 6.2 GitHub Release
- Create GitHub release with:
  - Model weights (.onnx files)
  - Manifest file
  - Training artifacts (config, metrics)
- Tag commit with `v0.1-wcag`

### 6.3 Model Registry
- Upload to HuggingFace Hub:
  - Repository: `turbolego/laya-wcag`
  - Include model card with training details
  - Add usage examples

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| 1. Create Kaggle Notebook | 1-2 days | [x] Initial scaffold |
| 2. Model Training | 2-3 days | Pending |
| 3. Evaluation | 1 day | Pending |
| 4. Export & Quantization | 1 day | Pending |
| 5. Repository Integration | 1 day | Pending |
| 6. Publishing | 0.5 days | Pending |

**Total estimated:** 6-8 days

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Insufficient training data | Low model accuracy | Use data augmentation, synthetic data generation |
| GPU time limits on Kaggle | Incomplete training | Optimize batch sizes, use gradient accumulation |
| Model size exceeds constraints | Slow browser loading | Aggressive quantization, consider smaller checkpoint |
| False positives in detection | User trust issues | Conservative thresholding, confidence calibration |

---

## Success Criteria

- [ ] Kaggle notebook created and runnable
- [ ] Model achieves >0.70 macro F1 on test set
- [ ] ONNX export successful, model loads in ONNX Runtime Web
- [ ] Integrated into layaForWeb demo
- [ ] Published to GitHub with release artifacts
- [ ] Documentation updated

---

## Current Status

**Notebook:** `/notebooks/laya_wcag_training.ipynb`
- Initial scaffold with dependencies and model loading
- Ready for data loading and training code

**Repository:** https://github.com/turbolego/layaForWeb

**Plan:** `docs/md/laya-wcag-training-plan.md`
