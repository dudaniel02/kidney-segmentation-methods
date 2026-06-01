# Kidney & Tumour Segmentation — Method Comparison

Comparative study of seven segmentation approaches on the
[KiTS19](https://kits19.grand-challenge.org/) kidney and tumour dataset.
Each method is implemented as a standalone project under its own subfolder.

## Methods

| Folder | Method | Approach | Input |
|--------|--------|----------|-------|
| [`3DNNUNET/`](3DNNUNET/) | nnU-Net v2 | Automated 3D pipeline with two-stage training | 3D patch |
| [`MedNeXt/`](MedNeXt/) | MedNeXt | ConvNeXt-style 3D encoder-decoder with deep supervision | 3D patch |
| [`TwoStageSeg/`](TwoStageSeg/) | Two-Stage Segmentation | Stage 1: coarse kidney localisation → Stage 2: fine tumour segmentation | 3D patch |
| [`UKAN/`](UKAN/) | U-KAN (3D) | 3D UNet with KAN channel-attention bottleneck — novel 3D application | 3D patch |
| [`VISTA3D/`](VISTA3D/) | VISTA3D | Fine-tuned VISTA3D foundation model (SegResNet, pretrained on 11 k CT volumes) | 3D patch |
| [`I-MedSAM/`](I-MedSAM/) | I-MedSAM | Frozen SAM ViT-B encoder + implicit neural representation decoder (2.5D) | 2.5D slice |
| [`dinov2/`](dinov2/) | DINOv2 | DINOv2 ViT features + lightweight segmentation head (2D) | 2D slice |

## Dataset — KiTS19

- 210 training cases, 90 test cases
- CT volumes with expert kidney and tumour annotations
- Labels: `0` background · `1` kidney · `2` tumour
- Source: [`kits19`](https://github.com/neheller/kits19) — place raw data at `~/kits19/`

## Repository Structure

Each method folder follows the same layout:

```
<Method>/
├── src/            # Training, inference, and evaluation scripts
├── configs/        # YAML configuration files (where applicable)
├── notebooks/      # Exploratory notebooks (where applicable)
└── requirements.txt
```

Data, model checkpoints, and outputs are excluded from version control
(see [`.gitignore`](.gitignore)).

## References

- **nnU-Net**: Isensee et al., *Nature Methods* 2021 — [arXiv:1809.10486](https://arxiv.org/abs/1809.10486)
- **MedNeXt**: Roy et al., *MICCAI 2023* — [arXiv:2303.09975](https://arxiv.org/abs/2303.09975)
- **U-KAN**: Li et al., *AAAI 2025* — [arXiv:2406.02918](https://arxiv.org/abs/2406.02918)
- **VISTA3D**: He et al., *CVPR 2025* — [arXiv:2406.05285](https://arxiv.org/abs/2406.05285)
- **I-MedSAM**: Wei et al., *ECCV 2024* — [arXiv:2311.17081](https://arxiv.org/abs/2311.17081)
- **SAM**: Kirillov et al., *ICCV 2023* — [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- **DINOv2**: Oquab et al., *TMLR 2024* — [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
