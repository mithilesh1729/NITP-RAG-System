import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def create_architecture_diagram(output_path="presentation/abc.png"):
    # 16:10 aspect ratio dark slate figure
    fig, ax = plt.subplots(figsize=(16, 10.5), facecolor="#141414")
    ax.set_facecolor("#141414")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    # Color definitions
    container_bg = "#1e1e1e"
    container_border = "#3e3e3e"
    box_bg = "#262626"
    box_border = "#555555"
    text_white = "#ffffff"
    text_muted = "#94a3b8"
    accent_cyan = "#38bdf8"
    accent_emerald = "#34d399"
    badge_bg = "#181818"

    # ==========================================
    # 1. PHASE 1: INGESTION CONTAINER (LEFT)
    # ==========================================
    p1_box = FancyBboxPatch((0.8, 2.4), 5.6, 7.4,
                            boxstyle="round,pad=0.1,rounding_size=0.15",
                            facecolor=container_bg, edgecolor=container_border, linewidth=1.2)
    ax.add_patch(p1_box)
    ax.text(1.1, 9.4, "Phase 1: Ingestion — ingest.py", fontsize=11.5, color="#cbd5e1", fontweight="semibold")

    # Ingestion Nodes (Centered at X = 3.6)
    # Node 1: Notices
    n1 = FancyBboxPatch((1.5, 8.1), 4.2, 0.85, boxstyle="round,pad=0.08,rounding_size=0.4",
                        facecolor=box_bg, edgecolor=box_border, linewidth=1.2)
    ax.add_patch(n1)
    ax.text(3.6, 8.52, "NITP PDF Notices", fontsize=11, color=text_white, ha="center", va="center", fontweight="medium")

    # Node 2: Parser
    n2 = FancyBboxPatch((1.2, 6.4), 4.8, 1.0, boxstyle="square,pad=0.08",
                        facecolor=box_bg, edgecolor=box_border, linewidth=1.2)
    ax.add_patch(n2)
    ax.text(3.6, 6.9, "PDF-to-Markdown Parser\n(pymupdf4llm)", fontsize=10.5, color=text_white, ha="center", va="center")

    # Node 3: Chunker
    n3 = FancyBboxPatch((1.2, 4.7), 4.8, 1.0, boxstyle="square,pad=0.08",
                        facecolor=box_bg, edgecolor=box_border, linewidth=1.2)
    ax.add_patch(n3)
    ax.text(3.6, 5.2, "Semantic Markdown Chunker\n(4000 chars)", fontsize=10.5, color=text_white, ha="center", va="center")

    # Node 4: Embedding Model
    n4 = FancyBboxPatch((1.0, 3.0), 5.2, 1.0, boxstyle="square,pad=0.08",
                        facecolor=box_bg, edgecolor=accent_cyan, linewidth=1.5)
    ax.add_patch(n4)
    ax.text(3.6, 3.5, "Cloud Embedding Model\n(gemini-embedding-2 / 3072 dim)", fontsize=10.5, color=text_white, ha="center", va="center", fontweight="semibold")

    # ==========================================
    # 2. PHASE 2: CHAT & RETRIEVAL CONTAINER (RIGHT)
    # ==========================================
    p2_box = FancyBboxPatch((7.6, 2.0), 7.6, 7.8,
                            boxstyle="round,pad=0.1,rounding_size=0.15",
                            facecolor=container_bg, edgecolor=container_border, linewidth=1.2)
    ax.add_patch(p2_box)
    ax.text(7.9, 9.4, "Phase 2: Chat & Retrieval — chat.py", fontsize=11.5, color="#cbd5e1", fontweight="semibold")

    # Node 5: Student Query
    n5 = FancyBboxPatch((9.5, 8.1), 4.0, 0.85, boxstyle="round,pad=0.08,rounding_size=0.4",
                        facecolor=box_bg, edgecolor=box_border, linewidth=1.2)
    ax.add_patch(n5)
    ax.text(11.5, 8.52, "Student Query", fontsize=11, color=text_white, ha="center", va="center", fontweight="medium")

    # Node 6: Embedding Model (Right side)
    n6 = FancyBboxPatch((10.3, 5.9), 4.5, 1.1, boxstyle="square,pad=0.08",
                        facecolor=box_bg, edgecolor=accent_cyan, linewidth=1.5)
    ax.add_patch(n6)
    ax.text(12.55, 6.45, "Cloud Embedding Model\n(gemini-embedding-2)", fontsize=10, color=text_white, ha="center", va="center", fontweight="semibold")

    # Node 7: LLM Generator (Left side of right container)
    n7 = FancyBboxPatch((8.0, 4.4), 3.8, 1.1, boxstyle="square,pad=0.08",
                        facecolor=box_bg, edgecolor=accent_emerald, linewidth=1.5)
    ax.add_patch(n7)
    ax.text(9.9, 4.95, "LLM Generator\n(Gemini Flash API)", fontsize=10.5, color=text_white, ha="center", va="center", fontweight="semibold")

    # Node 8: Answer + Source Citation
    n8 = FancyBboxPatch((7.8, 2.6), 4.2, 0.85, boxstyle="round,pad=0.08,rounding_size=0.4",
                        facecolor=box_bg, edgecolor=box_border, linewidth=1.2)
    ax.add_patch(n8)
    ax.text(9.9, 3.02, "Answer + Source Citation", fontsize=10.5, color=text_white, ha="center", va="center", fontweight="medium")

    # ==========================================
    # 3. CHROMADB VECTOR STORE (BOTTOM CENTER)
    # ==========================================
    db_box = FancyBboxPatch((3.5, 0.4), 4.6, 1.3, boxstyle="round,pad=0.1,rounding_size=0.25",
                            facecolor="#1e293b", edgecolor=accent_cyan, linewidth=1.8)
    ax.add_patch(db_box)
    ax.text(5.8, 1.18, "ChromaDB Vector Store", fontsize=11.5, color="#f8fafc", ha="center", va="center", fontweight="bold")
    ax.text(5.8, 0.75, "(Cosine Distance / 3072 Dimensions)", fontsize=9.5, color="#94a3b8", ha="center", va="center")

    # ==========================================
    # 4. ORTHOGONAL STEPPED ARROWS (NO CROSSINGS)
    # ==========================================
    # Ingestion vertical flow
    ax.annotate("", xy=(3.6, 7.4), xytext=(3.6, 8.1),
                arrowprops=dict(arrowstyle="-|>", color=accent_cyan, lw=1.6, mutation_scale=15))
    ax.annotate("", xy=(3.6, 5.7), xytext=(3.6, 6.4),
                arrowprops=dict(arrowstyle="-|>", color=accent_cyan, lw=1.6, mutation_scale=15))
    ax.annotate("", xy=(3.6, 4.0), xytext=(3.6, 4.7),
                arrowprops=dict(arrowstyle="-|>", color=accent_cyan, lw=1.6, mutation_scale=15))

    # Ingestion -> ChromaDB (Straight down)
    ax.annotate("", xy=(3.6, 1.7), xytext=(3.6, 3.0),
                arrowprops=dict(arrowstyle="-|>", color=accent_cyan, lw=1.6, mutation_scale=15))
    ax.text(3.6, 2.35, "Stores Vectors + Metadata", fontsize=8.5, color="#e2e8f0", ha="center",
            bbox=dict(facecolor=badge_bg, edgecolor="#475569", boxstyle="round,pad=0.25", lw=0.8))

    # Student Query -> Embedding Model (Right side)
    ax.plot([12.55, 12.55], [8.1, 7.0], color=accent_cyan, lw=1.6)
    ax.annotate("", xy=(12.55, 7.0), xytext=(12.55, 7.01),
                arrowprops=dict(arrowstyle="-|>", color=accent_cyan, lw=1.6, mutation_scale=15))

    # Embedding Model -> ChromaDB (Orthogonal stepped route along right perimeter)
    # Down along x=12.55 to y=1.05, then Left into right side of ChromaDB at (8.1, 1.05)
    ax.plot([12.55, 12.55, 8.1], [5.9, 1.05, 1.05], color=accent_cyan, lw=1.6)
    ax.annotate("", xy=(8.1, 1.05), xytext=(8.15, 1.05),
                arrowprops=dict(arrowstyle="-|>", color=accent_cyan, lw=1.6, mutation_scale=15))
    ax.text(10.8, 1.35, "Cosine Similarity Search", fontsize=8.5, color="#e2e8f0", ha="center",
            bbox=dict(facecolor=badge_bg, edgecolor="#475569", boxstyle="round,pad=0.25", lw=0.8))

    # ChromaDB -> LLM Generator (Orthogonal route through gap between containers)
    # From top of ChromaDB (x=6.9, y=1.7) up along x=6.9 to y=4.95, then Right into LLM at (8.0, 4.95)
    ax.plot([6.9, 6.9, 8.0], [1.7, 4.95, 4.95], color=accent_cyan, lw=1.6)
    ax.annotate("", xy=(8.0, 4.95), xytext=(7.95, 4.95),
                arrowprops=dict(arrowstyle="-|>", color=accent_cyan, lw=1.6, mutation_scale=15))
    ax.text(6.9, 3.3, "Top-10 Chunks\n+ Citations", fontsize=8, color="#e2e8f0", ha="center",
            bbox=dict(facecolor=badge_bg, edgecolor="#475569", boxstyle="round,pad=0.25", lw=0.8))

    # Student Query -> LLM Generator (Dashed straight vertical line)
    ax.plot([9.9, 9.9], [8.1, 5.5], color="#94a3b8", lw=1.3, ls="--")
    ax.annotate("", xy=(9.9, 5.5), xytext=(9.9, 5.55),
                arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.3, ls="--", mutation_scale=13))
    ax.text(9.9, 7.0, "Prompt Context", fontsize=8.5, color="#94a3b8", ha="center",
            bbox=dict(facecolor=badge_bg, edgecolor="#475569", boxstyle="round,pad=0.2", lw=0.8))

    # LLM Generator -> Answer + Citation (Straight vertical)
    ax.annotate("", xy=(9.9, 3.45), xytext=(9.9, 4.4),
                arrowprops=dict(arrowstyle="-|>", color=accent_cyan, lw=1.6, mutation_scale=15))

    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Diagram successfully updated and saved to {output_path}")

if __name__ == "__main__":
    create_architecture_diagram()
