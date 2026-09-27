"""
scripts/generate_pipeline_diagram.py
Generates a comprehensive, publication-quality architecture diagram
illustrating FloraGuard's complete end-to-end dataflow, backend integration,
bio-optical preprocessing, PyTorch inference engine, and agronomic intelligence.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "app" / "frontend" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "architectural_pipeline.png"

def create_pipeline_diagram():
    fig = plt.figure(figsize=(24, 11), facecolor='#080c14')
    ax = fig.add_axes([0, 0, 1, 1], facecolor='#080c14')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Title & Subtitle Banner
    ax.text(50, 95.8, "FloraGuard AI — End-to-End System Topology & Deep Learning Pipeline",
            fontsize=22, fontweight='bold', color='#ffffff', ha='center', va='center',
            family='sans-serif')
    ax.text(50, 92.4, "Full-Stack Dataflow: Client Ingestion ➔ FastAPI Gateway ➔ Bio-Optical Preprocessing ➔ PyTorch Accelerator ➔ 2-Tier Guardrails ➔ Agronomic Intelligence",
            fontsize=12, color='#94a3b8', ha='center', va='center', family='sans-serif')

    # Define 5 major stages with balanced spacing
    stages = [
        {
            "num": "01",
            "title": "Ingestion & API Gateway",
            "subtitle": "Client Capture & Async HTTP",
            "x": 2.5, "w": 17.0,
            "color": "#06b6d4",
            "bg_color": "#082f49",
            "border_color": "#0284c7",
            "badge": "FastAPI + httpx",
            "items": [
                ("User Inputs", "Leaf photo drag-drop or public image URL link"),
                ("FastAPI Endpoints", "POST /api/predict (multipart) & /api/predict-url"),
                ("Async Downloader", "httpx async client with 10MB streaming limit"),
                ("Buffer Decoder", "BytesIO stream to PIL Image in strict RGB mode"),
                ("Client Telemetry", "Records client latency & initiates processing state")
            ]
        },
        {
            "num": "02",
            "title": "Preprocessing & Gatekeeper",
            "subtitle": "Bio-Optical Filter & Tensors",
            "x": 22.0, "w": 17.0,
            "color": "#10b981",
            "bg_color": "#022c22",
            "border_color": "#059669",
            "badge": "Tier-1 OOD + TorchVision",
            "items": [
                ("Tier-1 Guardrail", "Bio-optical HSV chlorophyll green/chroma check"),
                ("OOD Rejection", "Instantly drops screenshots, UI code, non-foliage"),
                ("Spatial Transform", "Bicubic resize 256×256 -> CenterCrop to 224×224"),
                ("Tensor Conversion", "ToTensor() mapping [0, 255] float to [0.0, 1.0]"),
                ("Standardization", "ImageNet norm: μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]")
            ]
        },
        {
            "num": "03",
            "title": "Neural Inference Engine",
            "subtitle": "Dynamic Backbones & MPS/CUDA",
            "x": 41.5, "w": 17.0,
            "color": "#3b82f6",
            "bg_color": "#172554",
            "border_color": "#2563eb",
            "badge": "PyTorch + ModelManager",
            "items": [
                ("Device Dispatch", "Auto MPS (Apple Silicon), CUDA (NVIDIA), or CPU"),
                ("ModelManager", "Hot-swappable active architecture singleton"),
                ("MobileNetV3-Large", "Benchmark Winner (99.84% Acc, 4.25M params)"),
                ("ResNet-18", "Transfer learning residual backbone (99.83% Acc)"),
                ("Custom CNN & ViT", "Scratch CNN (541K params) & ViT-Base (85.8M)"),
                ("Forward Pass", "torch.no_grad() -> Logits -> Softmax probabilities")
            ]
        },
        {
            "num": "04",
            "title": "Confidence Gate & Agronomy",
            "subtitle": "Tier-2 Guardrail & Treatment",
            "x": 61.0, "w": 17.0,
            "color": "#f59e0b",
            "bg_color": "#451a03",
            "border_color": "#d97706",
            "badge": "Tier-2 OOD + Disease KB",
            "items": [
                ("Tier-2 Guardrail", "Neural confidence check (Flags OOD if Top-1 < 50%)"),
                ("Top-5 Distribution", "Calculates class probabilities & confidence spread"),
                ("Pathogen Mapping", "Maps 38 classes to Fungal, Bacterial, Viral, Healthy"),
                ("Cultural Controls", "Pruning, aeration, irrigation timing & sanitation"),
                ("Organic & Chemical", "Bio-fungicides, copper soaps, or targeted actives")
            ]
        },
        {
            "num": "05",
            "title": "Reactive Client UI",
            "subtitle": "Interactive Browser Dashboard",
            "x": 80.5, "w": 17.0,
            "color": "#a855f7",
            "bg_color": "#3b0764",
            "border_color": "#9333ea",
            "badge": "Vanilla JS + CSS",
            "items": [
                ("JSON Envelope", "Class name, common name, confidence & advice"),
                ("Diagnostic Badge", "High-contrast visual card with severity badge"),
                ("Confidence Meter", "Animated progress bar indicating certainty"),
                ("Probability Bars", "Top-5 differential diagnosis candidate breakdown"),
                ("Action Tabs", "Tabbed interface for Cultural, Organic & Chemical cures")
            ]
        }
    ]

    card_y = 12.5
    card_h = 74.0

    for i, stage in enumerate(stages):
        x = stage["x"]
        w = stage["w"]
        color = stage["color"]
        border = stage["border_color"]
        bg = stage["bg_color"]

        # Outer Card with subtle glow border
        card_box = patches.FancyBboxPatch(
            (x, card_y), w, card_h,
            boxstyle="round,pad=0.7,rounding_size=2.0",
            facecolor='#0b1120',
            edgecolor=border,
            linewidth=2.2,
            zorder=2
        )
        ax.add_patch(card_box)

        # Header Box
        header_h = 13.0
        header_y = card_y + card_h - header_h - 1.2
        header_box = patches.FancyBboxPatch(
            (x + 0.6, header_y), w - 1.2, header_h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            facecolor=bg,
            edgecolor=color,
            linewidth=1.4,
            zorder=3
        )
        ax.add_patch(header_box)

        # Step Number Badge
        ax.text(x + 2.0, header_y + header_h - 3.2, stage["num"],
                fontsize=11, fontweight='bold', color='#ffffff',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor=color, edgecolor='none'),
                ha='center', va='center', zorder=4)

        # Technology Pill Badge
        ax.text(x + w - 1.6, header_y + header_h - 3.2, stage["badge"],
                fontsize=7.8, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#080c14', edgecolor=color, linewidth=0.8),
                ha='right', va='center', zorder=4)

        # Title & Subtitle
        ax.text(x + 1.2, header_y + 5.8, stage["title"],
                fontsize=10.8, fontweight='bold', color='#ffffff',
                ha='left', va='center', zorder=4)
        ax.text(x + 1.2, header_y + 2.5, stage["subtitle"],
                fontsize=8.3, color='#94a3b8',
                ha='left', va='center', zorder=4)

        # Content Items
        item_start_y = header_y - 3.5
        y_step = 9.2

        for idx, (label, desc) in enumerate(stage["items"]):
            curr_y = item_start_y - (idx * y_step)

            # Sub-item Box
            item_box = patches.FancyBboxPatch(
                (x + 0.7, curr_y - 2.5), w - 1.4, y_step - 1.5,
                boxstyle="round,pad=0.3,rounding_size=0.8",
                facecolor=(1.0, 1.0, 1.0, 0.03),
                edgecolor=(1.0, 1.0, 1.0, 0.08),
                linewidth=0.8,
                zorder=3
            )
            ax.add_patch(item_box)

            # Bullet dot
            ax.plot(x + 1.6, curr_y + 1.8, marker='o', markersize=4.2, color=color, zorder=4)

            # Item Label
            ax.text(x + 2.5, curr_y + 1.8, label,
                    fontsize=8.5, fontweight='bold', color='#ffffff',
                    ha='left', va='center', zorder=4)

            # Item Description (wrapped)
            ax.text(x + 2.5, curr_y - 0.7, desc,
                    fontsize=7.6, color='#cbd5e1',
                    ha='left', va='center', zorder=4)

        # Connecting Arrow to Next Stage
        if i < len(stages) - 1:
            arrow_x_start = x + w + 0.3
            arrow_x_end = stages[i + 1]["x"] - 0.3
            arrow_y = card_y + (card_h / 2) + 2.0

            arrow = patches.FancyArrowPatch(
                (arrow_x_start, arrow_y),
                (arrow_x_end, arrow_y),
                arrowstyle='-|>',
                mutation_scale=14,
                color='#38bdf8',
                linewidth=2.2,
                zorder=5
            )
            ax.add_patch(arrow)

            # Dataflow label above arrow
            flow_labels = [
                "Raw Bytes",
                "Tensor",
                "Softmax",
                "Diagnosis"
            ]
            ax.text((arrow_x_start + arrow_x_end) / 2, arrow_y + 3.2, flow_labels[i],
                    fontsize=7.2, fontweight='bold', color='#38bdf8', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='#080c14', edgecolor='#0284c7', linewidth=0.6),
                    zorder=6)

    # Bottom Footer Strip: Architecture highlights
    footer_box = patches.FancyBboxPatch(
        (2.5, 3.2), 95.0, 6.2,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        facecolor='#0f172a',
        edgecolor='#1e293b',
        linewidth=1.2,
        zorder=2
    )
    ax.add_patch(footer_box)

    highlights = [
        "GUARDRAILS: 2-Tier Out-of-Distribution Protection (Bio-Optical Chlorophyll Filter + Neural Confidence Threshold)",
        "HARDWARE: Sub-30ms Inference on Apple Silicon (MPS) / NVIDIA (CUDA) via PyTorch ModelManager Singleton",
        "KNOWLEDGE BASE: Production Agronomic Treatment Database Covering 38 Pathogen Classes across 14 Crop Species"
    ]
    for idx, text in enumerate(highlights):
        hx = 4.5 + (idx * 31.8)
        ax.text(hx, 6.3, text, fontsize=7.6, color='#94a3b8', ha='left', va='center', zorder=3)

    plt.savefig(OUTPUT_FILE, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Generated enhanced architectural pipeline diagram: {OUTPUT_FILE}")

if __name__ == "__main__":
    create_pipeline_diagram()
