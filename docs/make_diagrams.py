"""Regenerate the project diagrams in docs/ (PNG + SVG).

    py docs/make_diagrams.py

Pure matplotlib — no graphviz / cairo needed. Boxes size themselves to their
wrapped text, so editing the copy below never overflows a border. Outputs:

  docs/architecture_high_level.{png,svg}
  docs/use_case_diagram.{png,svg}
"""

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent

# ── palette ───────────────────────────────────────────────────────────────────
INK      = "#0f172a"
MUTED    = "#64748b"
PANEL_BG = "#f8fafc"
PANEL_ED = "#cbd5e1"

CORE     = ("#eef2ff", "#6366f1")   # our own modules
EXTERNAL = ("#fef3c7", "#d97706")   # Jira / LLM vendors
STORE    = ("#ecfdf5", "#10b981")   # persisted state
CLIENT   = ("#f1f5f9", "#475569")   # actors / clients
ACCENT   = ("#fce7f3", "#db2777")   # the vectorless decision layer
APIC     = ("#e0e7ff", "#4338ca")

FONT = "DejaVu Sans"
LH    = 18          # px per wrapped line at ssize ≈ 9.3pt
PAD   = 52          # title + top/bottom padding inside a box


# ── text metrics (axes units are px, since dpi=100) ───────────────────────────
def wrap_at(w, ssize=9.3, gutter=46):
    """How many characters fit on one line inside a box `w` px wide."""
    return max(18, int((w - gutter) / (ssize * 0.80)))


def lines_for(sub, w, ssize=9.3):
    return textwrap.wrap(sub, wrap_at(w, ssize)) if sub else []


def height_for(sub, w, ssize=9.3):
    n = max(1, len(lines_for(sub, w, ssize)))
    return PAD + LH * n


# ── primitives ────────────────────────────────────────────────────────────────
def canvas(w, h):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200 if ext == "png" else None,
                    facecolor="white", bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(f"wrote docs/{name}.png + .svg")


def panel(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14",
                                fc=PANEL_BG, ec=PANEL_ED, lw=1.4, zorder=1))
    ax.text(x + 22, y + h - 26, title, ha="left", va="center", fontsize=12.5,
            color=MUTED, family=FONT, fontweight="bold", zorder=3)


class Box:
    def __init__(self, ax, x, y, w, h, title, sub="", style=CORE, tsize=11.5,
                 ssize=9.3):
        fc, ec = style
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0,rounding_size=9",
                                    fc=fc, ec=ec, lw=1.7, zorder=2))
        self.x, self.y, self.w, self.h = x, y, w, h
        self.cx, self.cy = x + w / 2, y + h / 2
        self.top, self.bot, self.left, self.right = y + h, y, x, x + w
        if sub:
            rows = lines_for(sub, w, ssize)
            ax.text(self.cx, y + h - 22, title, ha="center", va="center",
                    fontsize=tsize, color=INK, family=FONT, fontweight="bold",
                    zorder=3)
            ax.text(self.cx, y + 16 + LH * len(rows) / 2, "\n".join(rows),
                    ha="center", va="center", fontsize=ssize, color="#334155",
                    family=FONT, zorder=3, linespacing=1.4)
        else:
            ax.text(self.cx, self.cy, title, ha="center", va="center", fontsize=tsize,
                    color=INK, family=FONT, fontweight="bold", zorder=3)


def arrow(ax, p1, p2, style="-|>", color=INK, lw=1.7, dashed=False, rad=0.0,
          cstyle=None):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, zorder=4,
                                 linestyle=(0, (5, 4)) if dashed else "solid",
                                 connectionstyle=cstyle or f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


def note(ax, x, y, text, color=MUTED, size=9, italic=True):
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color,
            family=FONT, style="italic" if italic else "normal", zorder=6,
            bbox=dict(fc="white", ec="none", pad=2.0))


# ══════════════════════════════════════════════════════════════════════════════
# 1. High-level architecture
# ══════════════════════════════════════════════════════════════════════════════
LEFT_TOP = [
    ("Jira Cloud",
     "JQL: project SAC · Reporting Area = Order Fallout · Resolved / Closed",
     EXTERNAL),
    ("Uploaded workaround docs",
     ".docx / .pdf / .txt kept in trackers/workaround_docs", CORE),
]
LEFT = [
    ("ingest.py",
     "tickets: incremental sync on `updated`, pick the resolution comment "
     "(assignee → resolver → any human), parse Order Type / Order Reason, prune "
     "tickets that left the JQL   ·   docs: 800-char chunks, overlap 100", CORE),
    ("textclean.py",
     "strip ADF markup / signatures / noise, lift `=== FIX ===` blocks, flag pointer "
     "comments (\"duplicate of SAC-x\")", CORE),
    ("embedder.py",
     "all-MiniLM-L6-v2 — always local, no ticket data leaves the host", CORE),
    ("vectordb.py   ·   Chroma",
     "four collections: step + error, for tickets and for documents. A hit must match "
     "the FAILURE, not shared prose.", CORE),
]
LEFT_BOT = [
    ("retrieval.py", "BM25 keyword index", CORE),
    ("graph.py", "failure-signature + pointer-link graph", CORE),
]
LEFT_TAIL = (
    "scorecard.py   —   accuracy self-test",
    "after each ingest, replays labelled failures through the very same pipeline and "
    "logs Hit@1 / Hit@3 — no feedback data required", CORE,
)
RIGHT = [
    ("Query", "a Jira ticket ID (fetched live) or pasted failure text", CLIENT),
    ("search.parse_input",
     "labelled-field regex, LLM fallback for free prose  →  failed step · error · "
     "Order Type / Order Reason", CORE),
    ("vectordb.search_dual",
     "dual cosine on the step and error legs, error weighted 0.65 — the error is the "
     "differentiator, the step is shared by many tickets", CORE),
    ("+ hybrid & graph recall",
     "BM25 ⊕ vector fused with RRF (k=60), then sibling / pointer-linked tickets "
     "pulled in. Both LLM-free.", CORE),
    ("errormatch.py   —   lexical gate",
     "re-scores step + error exactly and DROPS any candidate whose error CODE "
     "contradicts the query: a different error is a different failure. This layer is "
     "where the accuracy comes from.", ACCENT),
    ("re-rank, then split at 0.70",
     "optional LLM relevance filter · feedback votes · Order Type/Reason match · "
     "resolution quality · recency  →  strong matches vs. weak context", CORE),
    ("suggest.py   →   watable.py",
     "the team's 8-field workaround table. Four rows come from THIS ticket in code; "
     "the LLM writes only Cause / Solution applied / System modified / Customer "
     "action — a clean source is shown verbatim instead.", CORE),
    ("Answer",
     "table + % match + source tickets  →  rendered in the UI or posted as a Jira "
     "comment. No match → an honest \"no workaround\".", STORE),
]


def architecture():
    W = 1840
    LX, LW = 40, 830
    RX, RW = 950, 830
    iw = LW - 60
    half = (iw - 26) / 2

    # measure both columns before sizing the canvas
    h_ltop = max(height_for(s, half) for _, s, _ in LEFT_TOP)
    h_left = [height_for(s, iw) for _, s, _ in LEFT]
    h_lbot = max(height_for(s, half) for _, s, _ in LEFT_BOT)
    h_tail = height_for(LEFT_TAIL[1], iw)
    left_h = h_ltop + sum(h_left) + h_lbot + h_tail + 32 * 6

    qw = RW - 60
    h_right = [height_for(s, qw) for _, s, _ in RIGHT]
    right_h = sum(h_right) + 18 * (len(RIGHT) - 1)

    PH = 62 + max(left_h, right_h) + 34            # pipeline panel height
    H  = 92 + 118 + 16 + 70 + 28 + PH + 22 + 118 + 34
    fig, ax = canvas(W, H)

    ax.text(W / 2, H - 36, "FalloutAssist — Workaround Finder", ha="center",
            va="center", fontsize=22, color=INK, family=FONT, fontweight="bold")
    ax.text(W / 2, H - 68, "High-level architecture   ·   retrieval-first: the "
            "vectorless layers decide, the LLM only writes the final table",
            ha="center", va="center", fontsize=12, color=MUTED, family=FONT)

    # clients & triggers
    ctop = H - 92
    panel(ax, 40, ctop - 118, W - 80, 118, "CLIENTS & TRIGGERS")
    cw, ch = 405, 70
    clients = [
        Box(ax, 70 + i * (cw + 27), ctop - 104, cw, ch, t, s, CLIENT, 11, 8.8)
        for i, (t, s) in enumerate([
            ("Browser UI", "templates/index.html — ask, rate, upload, settings"),
            ("Knowledge admin", "ingest, documents, LLM keys, scorecard"),
            ("Auto-ingest scheduler", "app.py background thread"),
            ("Jira auto-suggest bot", "jirabot.py poller — comments on live tickets"),
        ])]

    atop = ctop - 118 - 16
    api = Box(ax, 40, atop - 70, W - 80, 70, "app.py   —   FastAPI",
              "/api/ask      /api/ingest/*      /api/documents      /api/feedback      "
              "/api/suggestions/*      /api/llm/*      /api/jira-feedback      "
              "/api/scorecard", APIC, 13, 9.7)
    for b in clients:
        arrow(ax, (b.cx, b.bot), (b.cx, api.top), lw=1.5)

    PT = atop - 70 - 28
    PB = PT - PH
    panel(ax, LX, PB, LW, PH, "INDEXING PATH   (offline, incremental)")
    panel(ax, RX, PB, RW, PH, "QUERY PATH   (one failure at a time)")
    arrow(ax, (455, api.bot), (455, PT - 44), lw=1.6)
    arrow(ax, (1365, api.bot), (1365, PT - 44), lw=1.6)

    # left column
    ix = LX + 30
    y = PT - 62
    jira = Box(ax, ix, y - h_ltop, half, h_ltop, *LEFT_TOP[0][:2],
               LEFT_TOP[0][2], 11.5, 9.3)
    updocs = Box(ax, ix + half + 26, y - h_ltop, half, h_ltop, *LEFT_TOP[1][:2],
                 LEFT_TOP[1][2], 11.5, 9.3)
    y -= h_ltop + 32
    rows = []
    for (t, s, st), h in zip(LEFT, h_left):
        rows.append(Box(ax, ix, y - h, iw, h, t, s, st))
        y -= h + 32
    bm25 = Box(ax, ix, y - h_lbot, half, h_lbot, *LEFT_BOT[0][:2], LEFT_BOT[0][2])
    grph = Box(ax, ix + half + 26, y - h_lbot, half, h_lbot, *LEFT_BOT[1][:2],
               LEFT_BOT[1][2])
    y -= h_lbot + 32
    tail = Box(ax, ix, y - h_tail, iw, h_tail, *LEFT_TAIL[:2], LEFT_TAIL[2])
    arrow(ax, (bm25.cx, bm25.bot), (tail.cx - 170, tail.top), lw=1.5, dashed=True,
          color="#10b981")
    arrow(ax, (grph.cx, grph.bot), (tail.cx + 170, tail.top), lw=1.5, dashed=True,
          color="#10b981")

    arrow(ax, (jira.cx, jira.bot), (rows[0].cx - 170, rows[0].top), lw=1.6)
    arrow(ax, (updocs.cx, updocs.bot), (rows[0].cx + 170, rows[0].top), lw=1.6)
    for a, b in zip(rows, rows[1:]):
        arrow(ax, (a.cx, a.bot), (b.cx, b.top), lw=1.6)
    arrow(ax, (rows[-1].cx - 170, rows[-1].bot), (bm25.cx, bm25.top), lw=1.6)
    arrow(ax, (rows[-1].cx + 170, rows[-1].bot), (grph.cx, grph.top), lw=1.6)

    # right column
    qx = RX + 30
    y = PT - 62
    Q = []
    for (t, s, st), h in zip(RIGHT, h_right):
        Q.append(Box(ax, qx, y - h, qw, h, t, s, st))
        y -= h + 18
    for a, b in zip(Q, Q[1:]):
        arrow(ax, (a.cx, a.bot), (b.cx, b.top), lw=1.6)

    # the index feeds the query path
    arrow(ax, (rows[-1].right, rows[-1].cy), (Q[2].left, Q[2].cy), lw=1.6,
          dashed=True, color="#6366f1", rad=-0.14)
    arrow(ax, (grph.right, grph.cy), (Q[3].left, Q[3].cy - 14), lw=1.5,
          dashed=True, color="#6366f1", rad=-0.10)
    note(ax, 908, (rows[-1].cy + Q[2].cy) / 2 + 34, "reads\nthe index",
         color="#4338ca", size=8.8)

    # bottom: state + providers
    bt = PB - 22 - 118
    panel(ax, 40, bt, 1080, 118, "PERSISTED STATE   —   trackers/   (gitignored)")
    sw = (1080 - 5 * 22) / 4
    for i, (t, s) in enumerate([
            ("workaround_index/", "Chroma vectors"),
            ("workaround_docs/", "uploaded files"),
            ("feedback.json", "votes, scoped per failure"),
            ("ingest_state.json", "+ jira_suggestions.json"),
    ]):
        Box(ax, 62 + i * (sw + 22), bt + 14, sw, 70, t, s, STORE, 10.2, 8.5)

    panel(ax, 1150, bt, 690, 118, "LLM PROVIDERS   —   generate.py      "
          "(synthesis / re-rank only — never retrieval)")
    Box(ax, 1172, bt + 14, 320, 70, "Synapt (Azure OpenAI)",
        "governed · the active provider", EXTERNAL, 10.2, 8.5)
    Box(ax, 1510, bt + 14, 308, 70, "Groq · Gemini · Claude · Ollama",
        "swappable · key rotation + cooldown", EXTERNAL, 10.2, 8.5)

    arrow(ax, (1810, bt + 118), (Q[6].right, Q[6].cy), lw=1.7, dashed=True,
          color="#d97706", cstyle="angle,angleA=90,angleB=0,rad=10")

    ax.text(W / 2, 18, "Every retrieval and ranking layer is LLM-free; scorecard.py "
            "replays labelled failures after each ingest to keep Hit@1 honest.",
            ha="center", va="center", fontsize=10, color=MUTED, family=FONT,
            style="italic")

    save(fig, "architecture_high_level")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Use case diagram
# ══════════════════════════════════════════════════════════════════════════════
def stick_actor(ax, x, y, label, sub=""):
    ax.add_patch(Circle((x, y + 46), 15, fc="white", ec=INK, lw=2, zorder=3))
    ax.plot([x, x], [y + 31, y - 4], color=INK, lw=2, zorder=3)
    ax.plot([x - 22, x + 22], [y + 20, y + 20], color=INK, lw=2, zorder=3)
    ax.plot([x, x - 20], [y - 4, y - 36], color=INK, lw=2, zorder=3)
    ax.plot([x, x + 20], [y - 4, y - 36], color=INK, lw=2, zorder=3)
    ax.text(x, y - 56, label, ha="center", va="center", fontsize=12, color=INK,
            family=FONT, fontweight="bold", zorder=3)
    if sub:
        ax.text(x, y - 76, sub, ha="center", va="center", fontsize=9, color=MUTED,
                family=FONT, zorder=3)


def sys_actor(ax, x, y, label, sub=""):
    w, h = 200, 62
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=8",
                                fc=EXTERNAL[0], ec=EXTERNAL[1], lw=1.8, zorder=3))
    ax.text(x, y + 16, "«system»", ha="center", va="center", fontsize=8.5,
            color="#b45309", family=FONT, style="italic", zorder=4)
    ax.text(x, y - 1, label, ha="center", va="center", fontsize=11.5, color=INK,
            family=FONT, fontweight="bold", zorder=4)
    if sub:
        ax.text(x, y - 19, sub, ha="center", va="center", fontsize=8.6,
                color="#475569", family=FONT, zorder=4)
    return dict(l=x - w / 2, r=x + w / 2, t=y + h / 2, b=y - h / 2)


class UseCase:
    def __init__(self, ax, cx, cy, text, rx=170, ry=37, style=CORE):
        fc, ec = style
        ax.add_patch(Ellipse((cx, cy), rx * 2, ry * 2, fc=fc, ec=ec, lw=1.8, zorder=2))
        ax.text(cx, cy, "\n".join(textwrap.wrap(text, 27)), ha="center", va="center",
                fontsize=10.4, color=INK, family=FONT, zorder=3, linespacing=1.3)
        self.cx, self.cy, self.rx, self.ry = cx, cy, rx, ry

    def edge(self, tx, ty):
        dx, dy = tx - self.cx, ty - self.cy
        n = ((dx / self.rx) ** 2 + (dy / self.ry) ** 2) ** 0.5 or 1
        return (self.cx + dx / n, self.cy + dy / n)


def assoc(ax, uc, p):
    """UML association — a plain line between an actor and a use case."""
    ax.plot(*zip(uc.edge(*p), p), color="#334155", lw=1.5, zorder=1,
            solid_capstyle="round")


def route(ax, uc, waypoints):
    """Association drawn as a right-angled rail, to dodge other elements."""
    pts = list(waypoints) + [uc.edge(*waypoints[-1])]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#334155", lw=1.5,
            zorder=1, solid_capstyle="round", solid_joinstyle="round")


def stereotype(ax, src, dst, kind="include", label_at=None, rad=0.0):
    p1 = src.edge(dst.cx, dst.cy)
    p2 = dst.edge(src.cx, src.cy)
    arrow(ax, p1, p2, color="#6366f1", lw=1.4, dashed=True, rad=rad)
    lx, ly = label_at or ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    ax.text(lx, ly, f"«{kind}»", ha="center", va="center", fontsize=8.8,
            color="#4338ca", family=FONT, style="italic", zorder=6,
            bbox=dict(fc="white", ec="none", pad=1.6))


def use_case():
    W, H = 1790, 1200
    fig, ax = canvas(W, H)

    ax.text(W / 2, H - 32, "FalloutAssist — Use Case Diagram", ha="center", va="center",
            fontsize=22, color=INK, family=FONT, fontweight="bold")
    ax.text(W / 2, H - 62, "Salesforce order-fallout support  ·  who does what with the "
            "workaround finder", ha="center", va="center", fontsize=12, color=MUTED,
            family=FONT)

    bx, by, bw, bh = 425, 58, 790, 1032
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                boxstyle="round,pad=0,rounding_size=16",
                                fc="#fbfcfe", ec="#475569", lw=2, zorder=0))
    ax.text(bx + bw / 2, by + bh - 26, "FalloutAssist   (app.py)", ha="center",
            va="center", fontsize=13.5, color="#334155", family=FONT,
            fontweight="bold", zorder=1)

    cA, cB = 610, 1015

    # column A — human-initiated
    find = UseCase(ax, cA, 998, "Find a workaround for a failed step + error")
    byid = UseCase(ax, cA, 893, "Ask by Jira ticket ID (fetched live)")
    rate = UseCase(ax, cA, 788, "Rate a suggestion (up / down)")
    subm = UseCase(ax, cA, 683, "Submit a workaround of my own")
    revw = UseCase(ax, cA, 578, "Review & approve user-submitted fixes")
    upld = UseCase(ax, cA, 473, "Upload a workaround document")
    mdoc = UseCase(ax, cA, 368, "Manage the indexed documents")
    conf = UseCase(ax, cA, 263, "Configure the LLM provider & keys")
    card = UseCase(ax, cA, 158, "Check the accuracy scorecard")

    # column B — automated / system-facing
    post = UseCase(ax, cB, 998, "Auto-post a suggestion on a live ticket")
    rank = UseCase(ax, cB, 868, "Retrieve & rank past resolutions", style=ACCENT)
    synt = UseCase(ax, cB, 738, "Synthesize the 8-field workaround table")
    harv = UseCase(ax, cB, 608, "Harvest in-Jira feedback")
    bild = UseCase(ax, cB, 478, "Build the BM25 + failure-graph indexes")
    ingt = UseCase(ax, cB, 348, "Ingest resolved tickets from Jira")

    # actors
    stick_actor(ax, 190, 845, "Support Engineer", "L1 / L2 fallout desk")
    stick_actor(ax, 190, 368, "Knowledge Admin", "owns the index & config")
    stick_actor(ax, 1430, 985, "Scheduler", "background timers")
    jira = sys_actor(ax, 1430, 780, "Jira Cloud", "tickets, comments, fields")
    llm  = sys_actor(ax, 1430, 600, "LLM Provider", "Synapt / Groq / Ollama")
    emb  = sys_actor(ax, 1430, 400, "Local Embedder", "all-MiniLM-L6-v2")

    for uc, p in [(find, (250, 890)), (byid, (250, 868)), (rate, (250, 825)),
                  (subm, (250, 805))]:
        assoc(ax, uc, p)
    for uc, p in [(revw, (250, 400)), (upld, (250, 382)), (mdoc, (250, 368)),
                  (conf, (250, 354)), (card, (250, 336))]:
        assoc(ax, uc, p)
    # manual "ingest now" — railed along the bottom so it crosses nothing
    route(ax, ingt, [(250, 352), (310, 352), (310, 96), (cB, 96)])
    note(ax, 660, 108, "run an ingest now", size=8.6)

    # system actors
    assoc(ax, post, (jira["l"], 800))
    assoc(ax, harv, (jira["l"], 760))
    assoc(ax, synt, (llm["l"], 615))
    assoc(ax, rank, (llm["l"], 585))
    assoc(ax, ingt, (emb["l"], 386))
    assoc(ax, post, (1430, 950))
    route(ax, ingt, [(jira["r"], 780), (1545, 780), (1545, 348)])      # Jira → ingest
    route(ax, rank, [(emb["r"], 415), (1580, 415), (1580, 868)])       # embed the query
    route(ax, ingt, [(1466, 950), (1650, 950), (1650, 318)])           # timer → ingest

    # relationships
    stereotype(ax, find, rank, "include", label_at=(812, 950))
    stereotype(ax, byid, find, "include", label_at=(543, 945))
    stereotype(ax, post, find, "include", label_at=(812, 1012))
    stereotype(ax, synt, find, "extend", rad=0.16, label_at=(872, 934))
    stereotype(ax, harv, rate, "include", label_at=(812, 715))
    stereotype(ax, subm, revw, "include", label_at=(676, 630))
    stereotype(ax, ingt, bild, "include", label_at=(1082, 413))

    ax.text(W / 2, 24, "«include» = always part of that flow    ·    «extend» = optional "
            "(with the LLM off the matched resolution is shown verbatim — retrieval and "
            "ranking never need a model)",
            ha="center", va="center", fontsize=10, color=MUTED, family=FONT,
            style="italic")

    save(fig, "use_case_diagram")


if __name__ == "__main__":
    architecture()
    use_case()
