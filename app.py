import math
import io
import os
import base64
import numpy as np
from math import sqrt, radians, degrees, acos, cos, sin, tan, atan2
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Image as RLImage, KeepTogether
)
from trajectory import (
    calculate_all_wells,
    add_depth_dependent_uncertainty,
    error_ellipse_at_station,
    select_ellipsoid_indices,
    preliminary_anti_collision,
    error_envelope_tube,
    interpolated_uncertainty_path,
    error_envelope_tube_from_path,
)



# ----------------------------------------------------------------------
# PDF REPORT GENERATOR
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_LOGO = os.path.join(BASE_DIR, "rigsis_logo.png")
FAVICON_PATH = os.path.join(BASE_DIR, "anti_collision_icon.png")


def _fmt(v, decimals=2):
    try:
        x = float(v)
        if not np.isfinite(x):
            return "∞"
        return f"{x:,.{decimals}f}"
    except Exception:
        return str(v)


def _pdf_table(data, widths=None, header=True, font_size=7.5, status_col=None):
    tbl = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8B8B8")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B2B2B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    if status_col is not None and len(data) > 1:
        for r in range(1, len(data)):
            raw = data[r][status_col]
            status = raw.getPlainText() if hasattr(raw, "getPlainText") else str(raw)
            if "ENVELOPE OVERLAP" in status:
                style += [("BACKGROUND", (status_col, r), (status_col, r), colors.HexColor("#F4CCCC")),
                          ("TEXTCOLOR", (status_col, r), (status_col, r), colors.HexColor("#C00000")),
                          ("FONTNAME", (status_col, r), (status_col, r), "Helvetica-Bold")]
            elif "CRITICAL" in status:
                style += [("BACKGROUND", (status_col, r), (status_col, r), colors.HexColor("#FCE5CD")),
                          ("TEXTCOLOR", (status_col, r), (status_col, r), colors.HexColor("#E69138")),
                          ("FONTNAME", (status_col, r), (status_col, r), "Helvetica-Bold")]
            elif "CLOSE" in status:
                style += [("BACKGROUND", (status_col, r), (status_col, r), colors.HexColor("#FFF2CC")),
                          ("TEXTCOLOR", (status_col, r), (status_col, r), colors.HexColor("#BF9000")),
                          ("FONTNAME", (status_col, r), (status_col, r), "Helvetica-Bold")]
            elif "PRELIMINARY PASS" in status:
                style += [("BACKGROUND", (status_col, r), (status_col, r), colors.HexColor("#D9EAD3")),
                          ("TEXTCOLOR", (status_col, r), (status_col, r), colors.HexColor("#38761D")),
                          ("FONTNAME", (status_col, r), (status_col, r), "Helvetica-Bold")]
    tbl.setStyle(TableStyle(style))
    return tbl


def _figure_to_rl(fig, width_mm=175):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = RLImage(buf)
    img.drawWidth = width_mm * mm
    # Preserve aspect ratio.
    from PIL import Image as PILImage
    with PILImage.open(buf) as im:
        w, h = im.size
    img.drawHeight = img.drawWidth * h / w
    return img


def _plot_new_well(p3_detailed):
    fig, ax = plt.subplots(figsize=(7.0, 8.0))
    ax.plot(p3_detailed["Displacement"], p3_detailed["TVD"], linewidth=2.5, color="#ff4b4b", label="New Well")
    target_tvd = float(p3_detailed["TVD"].iloc[-1])
    target_disp = float(p3_detailed["Displacement"].iloc[-1])
    ax.scatter([target_disp], [target_tvd], marker="x", s=90, linewidths=2, color="#19d3c5", label="Target")
    max_tvd = max(float(p3_detailed["TVD"].max()), target_tvd, 1.0)
    max_disp = max(float(p3_detailed["Displacement"].max()), target_disp, 1.0)
    # Add a small margin so the full trajectory is never clipped by the axes.
    y_top = max_tvd * 1.04
    x_right = max_disp * 1.04
    # Keep a small left-side margin so the vertical section does not sit directly on the Y-axis.
    ax.set_xlim(-50, x_right)
    ax.set_ylim(y_top, 0)
    ax.set_title("VERTICAL VIEW", fontweight="bold")
    ax.set_xlabel("Displacement (m)")
    ax.set_ylabel("TVD (m)")
    ax.grid(True, linestyle=":", alpha=0.65)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def _plot_plan_new_vs_offset(offset_result, p3_detailed, color_map, surface_e, surface_n, target_e, target_n):
    fig, ax = plt.subplots(figsize=(8, 7))
    for well, odf in offset_result.groupby("Well"):
        ax.plot(odf["X"] - surface_e, odf["Y"] - surface_n,
                linewidth=1.8, color=color_map.get(str(well)), label=str(well))
    ax.plot(p3_detailed["E+"], p3_detailed["N+"], linewidth=3, color="#ff4b4b", label="New Well")
    ax.scatter([0], [0], color="#2ecc71", s=50, label="New Well Surface")
    ax.scatter([target_e - surface_e], [target_n - surface_n], color="#19d3c5", marker="x", s=90, linewidths=2, label="Target")
    ax.set_title("PLAN VIEW — NEW WELL vs OFFSET WELLS", fontweight="bold")
    ax.set_xlabel("ΔEasting from New Well Surface (m)")
    ax.set_ylabel("ΔNorthing from New Well Surface (m)")
    ax.grid(True, linestyle=":", alpha=0.65)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_static_3d(uresult, critical_row, color_map, surface_e, surface_n, surface_z):
    """Static trajectory-only 3D view for PDF report (Matplotlib).

    This intentionally remains independent from Kaleido. The trajectory-only
    report figure keeps the original Matplotlib implementation, while the
    error-envelope figure below is exported from the Phase 3 Plotly figure.
    """
    fig = plt.figure(figsize=(8.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    new = uresult[uresult["Well"].astype(str).str.strip().str.lower() == "new well"]
    offs = uresult[uresult["Well"].astype(str).str.strip().str.lower() != "new well"]

    for well, odf in offs.groupby("Well"):
        odf = odf.sort_values("MD")
        ax.plot(
            odf["X"] - surface_e,
            odf["Y"] - surface_n,
            odf["Z"] - surface_z,
            linewidth=1.5,
            color=color_map.get(str(well)),
            label=str(well),
        )

    new = new.sort_values("MD")
    ax.plot(
        new["X"] - surface_e,
        new["Y"] - surface_n,
        new["Z"] - surface_z,
        linewidth=2.8,
        color="#ff4b4b",
        label="New Well",
    )

    cx = [float(critical_row["New X"]) - surface_e, float(critical_row["Offset X"]) - surface_e]
    cy = [float(critical_row["New Y"]) - surface_n, float(critical_row["Offset Y"]) - surface_n]
    cz = [float(critical_row["New Z"]) - surface_z, float(critical_row["Offset Z"]) - surface_z]
    ax.plot(cx, cy, cz, color="#111111", linewidth=2.5, linestyle="--", label="Critical Separation")
    ax.scatter(cx, cy, cz, color="#111111", s=28)

    ax.set_xlabel("ΔEasting (m)")
    ax.set_ylabel("ΔNorthing (m)")
    ax.set_zlabel("ΔElevation (m)")
    ax.set_title("3D VIEW — PRELIMINARY ANTI-COLLISION", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def _plot_plotly_3d_anti_collision(uresult, selected, selected_well, color_map,
                                   surface_e, surface_n, surface_z):
    """Build the same Plotly 3D view used in Phase 3 for PDF export.

    The report intentionally reuses the Phase 3 coordinate system and visual
    logic: local coordinates are referenced to the New Well surface, and the
    same continuous uncertainty tubes are rendered from the same trajectory
    data and 5 m interpolation.
    """
    fig = go.Figure()

    new_ref = uresult[
        uresult["Well"].astype(str).str.strip().str.lower() == "new well"
    ].copy()
    offset_ref = uresult[
        uresult["Well"].astype(str).str.strip().str.lower() != "new well"
    ].copy()

    # Offset trajectories.
    for well, odf in offset_ref.groupby("Well"):
        odf = odf.sort_values("MD")
        fig.add_trace(go.Scatter3d(
            x=(odf["X"] - surface_e).astype(float),
            y=(odf["Y"] - surface_n).astype(float),
            z=(odf["Z"] - surface_z).astype(float),
            mode="lines",
            name=str(well),
            line=dict(width=4, color=color_map.get(str(well))),
            hovertemplate=(
                f"<b>{well}</b><br>"
                "MD=%{customdata[0]:.1f} m<br>"
                "TVDSS=%{customdata[1]:.1f} m<br>"
                "ΔEasting=%{x:.2f} m<br>"
                "ΔNorthing=%{y:.2f} m<extra></extra>"
            ),
            customdata=odf[["MD", "Z"]].astype(float).to_numpy(),
        ))

    # New Well trajectory.
    new_plot = new_ref.sort_values("MD")
    fig.add_trace(go.Scatter3d(
        x=(new_plot["X"] - surface_e).astype(float),
        y=(new_plot["Y"] - surface_n).astype(float),
        z=(new_plot["Z"] - surface_z).astype(float),
        mode="lines",
        name="New Well",
        line=dict(width=7, color="#ff4b4b"),
        hovertemplate=(
            "<b>New Well</b><br>"
            "MD=%{customdata[0]:.1f} m<br>"
            "TVDSS=%{customdata[1]:.1f} m<br>"
            "ΔEasting=%{x:.2f} m<br>"
            "ΔNorthing=%{y:.2f} m<extra></extra>"
        ),
        customdata=new_plot[["MD", "Z"]].astype(float).to_numpy(),
    ))

    # Continuous positional-error envelope tubes — exactly the same method as Phase 3.
    tube_sources = [("New Well", new_ref)] + [
        (str(w), g.copy()) for w, g in offset_ref.groupby("Well")
    ]
    tube_colors = {"New Well": "#ff4b4b", **color_map}
    for tube_name, tube_df in tube_sources:
        path = interpolated_uncertainty_path(tube_df, step_md=5.0)
        if len(path) < 2:
            continue
        verts, ti, tj, tk, _ = error_envelope_tube_from_path(path, n_ring=16)
        if not verts:
            continue
        fig.add_trace(go.Mesh3d(
            x=[v[0] - surface_e for v in verts],
            y=[v[1] - surface_n for v in verts],
            z=[v[2] - surface_z for v in verts],
            i=ti, j=tj, k=tk,
            name=f"{tube_name} Error Envelope",
            legendgroup=f"envelope_{tube_name}",
            showlegend=False,
            color=tube_colors.get(tube_name, "#999999"),
            opacity=0.16,
            hoverinfo="skip",
        ))

    # Critical separation line.
    nx = float(selected["New X"] - surface_e)
    ny = float(selected["New Y"] - surface_n)
    nz = float(selected["New Z"] - surface_z)
    ox = float(selected["Offset X"] - surface_e)
    oy = float(selected["Offset Y"] - surface_n)
    oz = float(selected["Offset Z"] - surface_z)
    tvdss = float(selected["TVDSS (m)"])
    ctc = float(selected["Center Distance (m)"])

    fig.add_trace(go.Scatter3d(
        x=[nx, ox], y=[ny, oy], z=[nz, oz],
        mode="lines",
        name="Critical Separation",
        line=dict(width=7, dash="dash", color="#111111"),
        hovertemplate=(
            f"<b>Critical Separation</b><br>"
            f"TVDSS={tvdss:.2f} m<br>"
            f"Distance={ctc:.2f} m<extra></extra>"
        ),
    ))

    # Critical point markers.
    fig.add_trace(go.Scatter3d(
        x=[nx], y=[ny], z=[nz], mode="markers",
        name="New Well Critical Point",
        showlegend=False,
        marker=dict(size=8, color="#ff4b4b", symbol="diamond"),
    ))
    fig.add_trace(go.Scatter3d(
        x=[ox], y=[oy], z=[oz], mode="markers",
        name="Offset Critical Point",
        showlegend=False,
        marker=dict(size=8, color="#111111", symbol="diamond"),
    ))

    # Critical error ellipses.
    from trajectory import error_ellipse_from_row
    new_row = {
        "X": float(selected["New X"]), "Y": float(selected["New Y"]), "Z": float(selected["New Z"]),
        "_tx": float(selected["New TX"]), "_ty": float(selected["New TY"]), "_tz": float(selected["New TZ"]),
        "Azimuth Error": float(selected["New Well Azimuth Error (m)"]),
        "Inclination Error": float(selected["New Well Inclination Error (m)"]),
    }
    off_row = {
        "X": float(selected["Offset X"]), "Y": float(selected["Offset Y"]), "Z": float(selected["Offset Z"]),
        "_tx": float(selected["Offset TX"]), "_ty": float(selected["Offset TY"]), "_tz": float(selected["Offset TZ"]),
        "Azimuth Error": float(selected["Offset Well Azimuth Error (m)"]),
        "Inclination Error": float(selected["Offset Well Inclination Error (m)"]),
    }
    new_ellipse = error_ellipse_from_row(new_row, n_points=48)
    off_ellipse = error_ellipse_from_row(off_row, n_points=48)

    for pts, name, color in [
        (new_ellipse, "New Well Error Envelope", "#ff4b4b"),
        (off_ellipse, f"{selected_well} Error Envelope", color_map.get(selected_well, "#999999")),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[p[0] - surface_e for p in pts],
            y=[p[1] - surface_n for p in pts],
            z=[p[2] - surface_z for p in pts],
            mode="lines",
            name=name,
            showlegend=False,
            line=dict(width=4, color=color),
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=dict(
            text="6.2  3D VIEW — PRELIMINARY ANTI-COLLISION WITH ERROR ENVELOPE",
            x=0, xanchor="left", font=dict(size=18),
        ),
        height=760,
        margin=dict(l=0, r=0, t=85, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.03,
            xanchor="left", x=0, font=dict(size=10),
            bgcolor="rgba(255,255,255,0.85)", borderwidth=0,
            entrywidth=110, itemsizing="constant",
        ),
        scene=dict(
            xaxis_title="ΔEasting (m)",
            yaxis_title="ΔNorthing (m)",
            zaxis_title="TVDSS relative to New Well Surface (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=1.55, z=1.25)),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _plotly_figure_to_rl(fig, width_mm=165):
    """Export a Plotly figure to PNG for embedding in the PDF report."""
    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=2)
    buf.seek(0)
    img = RLImage(buf)
    img.drawWidth = width_mm * mm
    from PIL import Image as PILImage
    with PILImage.open(buf) as im:
        w, h = im.size
    img.drawHeight = img.drawWidth * h / w
    return img

def _plot_ctc_sf(phase3_details, color_map, metric="ctc"):
    df = phase3_details.copy()
    if metric == "ctc":
        ycol, ylabel, title = "Center Distance (m)", "CtC (m)", "CENTER-TO-CENTER DISTANCE"
        ymin, ymax = 0, 100
    else:
        ycol, ylabel, title = "Separation Factor", "SF", "SEPARATION FACTOR"
        ymin, ymax = 0, 10
    df["New Well MD (m)"] = pd.to_numeric(df["New Well MD (m)"], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    df = df.dropna(subset=["New Well MD (m)", ycol])
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for well, g in df.groupby("Offset Well", sort=False):
        g = g.sort_values("New Well MD (m)")
        ax.plot(g["New Well MD (m)"], g[ycol], linewidth=2, color=color_map.get(str(well)), label=str(well))
    if metric == "ctc":
        row = df.loc[df[ycol].idxmin()]
        label = f"Min CtC = {float(row[ycol]):.2f} m"
    else:
        row = df.loc[df[ycol].idxmin()]
        label = f"Min SF = {float(row[ycol]):.2f}"
        ax.axhline(1.0, linestyle="--", linewidth=1.2, color="#d62728", label="SF = 1.0")
        ax.axhline(1.5, linestyle="--", linewidth=1.2, color="#ff7f0e", label="SF = 1.5")
        ax.axhline(3.0, linestyle="--", linewidth=1.2, color="#2ca02c", label="SF = 3.0")
    ax.scatter([row["New Well MD (m)"]], [row[ycol]], s=55, color=color_map.get(str(row["Offset Well"])), zorder=5)
    ax.annotate(label, (row["New Well MD (m)"], row[ycol]), xytext=(8, 8), textcoords="offset points", fontsize=9, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("New Well MD (m)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=7, ncol=2, loc="upper center")
    fig.tight_layout()
    return fig


def _plot_ctc_sf_combined(phase3_details, color_map):
    """Create the report CtC and SF plots as one guaranteed side-by-side figure."""
    df = phase3_details.copy()
    df["New Well MD (m)"] = pd.to_numeric(df["New Well MD (m)"], errors="coerce")
    df["Center Distance (m)"] = pd.to_numeric(df["Center Distance (m)"], errors="coerce")
    df["Separation Factor"] = pd.to_numeric(df["Separation Factor"], errors="coerce")
    df = df.dropna(subset=["New Well MD (m)"])

    fig, (ax_ctc, ax_sf) = plt.subplots(2, 1, figsize=(8.0, 9.8))

    for well, g in df.dropna(subset=["Center Distance (m)"]).groupby("Offset Well", sort=False):
        g = g.sort_values("New Well MD (m)")
        color = color_map.get(str(well))
        ax_ctc.plot(g["New Well MD (m)"], g["Center Distance (m)"],
                    linewidth=1.8, color=color, label=str(well))

    ctc_row = df.dropna(subset=["Center Distance (m)"]).loc[
        df.dropna(subset=["Center Distance (m)"])["Center Distance (m)"].idxmin()
    ]
    ctc_color = color_map.get(str(ctc_row["Offset Well"]))
    ax_ctc.scatter([ctc_row["New Well MD (m)"]], [ctc_row["Center Distance (m)"]],
                   s=48, color=ctc_color, zorder=5)
    ax_ctc.annotate(
        f"Min CtC = {float(ctc_row['Center Distance (m)']):.2f} m",
        (ctc_row["New Well MD (m)"], ctc_row["Center Distance (m)"]),
        xytext=(7, 7), textcoords="offset points", fontsize=8.5, fontweight="bold"
    )
    ax_ctc.set_title("CENTER-TO-CENTER DISTANCE", fontweight="bold", fontsize=11)
    ax_ctc.set_xlabel("New Well MD (m)", fontsize=9)
    ax_ctc.set_ylabel("CtC (m)", fontsize=9)
    ax_ctc.set_ylim(0, 100)
    ax_ctc.grid(True, linestyle=":", alpha=0.6)
    ax_ctc.legend(fontsize=7, ncol=2, loc="upper center")

    for well, g in df.dropna(subset=["Separation Factor"]).groupby("Offset Well", sort=False):
        g = g.sort_values("New Well MD (m)")
        color = color_map.get(str(well))
        ax_sf.plot(g["New Well MD (m)"], g["Separation Factor"],
                   linewidth=1.8, color=color, label=str(well))

    sf_df = df.dropna(subset=["Separation Factor"])
    sf_row = sf_df.loc[sf_df["Separation Factor"].idxmin()]
    sf_color = color_map.get(str(sf_row["Offset Well"]))
    ax_sf.scatter([sf_row["New Well MD (m)"]], [sf_row["Separation Factor"]],
                  s=48, color=sf_color, zorder=5)
    ax_sf.annotate(
        f"Min SF = {float(sf_row['Separation Factor']):.2f}",
        (sf_row["New Well MD (m)"], sf_row["Separation Factor"]),
        xytext=(7, 7), textcoords="offset points", fontsize=8.5, fontweight="bold"
    )
    ax_sf.axhline(1.0, linestyle="--", linewidth=1.1, color="#d62728", label="SF = 1.0")
    ax_sf.axhline(1.5, linestyle="--", linewidth=1.1, color="#ff7f0e", label="SF = 1.5")
    ax_sf.axhline(3.0, linestyle="--", linewidth=1.1, color="#2ca02c", label="SF = 3.0")
    ax_sf.set_title("SEPARATION FACTOR", fontweight="bold", fontsize=11)
    ax_sf.set_xlabel("New Well MD (m)", fontsize=9)
    ax_sf.set_ylabel("SF", fontsize=9)
    ax_sf.set_ylim(0, 10)
    ax_sf.grid(True, linestyle=":", alpha=0.6)
    ax_sf.legend(fontsize=7, ncol=2, loc="upper center")

    fig.tight_layout(h_pad=2.0)
    return fig

def _normalize_screening_status(status):
    text = str(status).strip()
    if "ENVELOPE OVERLAP" in text:
        return "ENVELOPE OVERLAP - MODIFY TRAJECTORY"
    if "CRITICAL" in text:
        return "CRITICAL - CONSIDER TO MODIFY TRAJECTORY"
    if "CLOSE" in text:
        return "CLOSE - REVIEW TRAJECTORY"
    if "PRELIMINARY PASS" in text:
        return "PRELIMINARY PASS"
    return text


def generate_pdf_report(project, result, p3_detailed, uncertainty_result, phase3_summary, phase3_details, include_survey_attachment=True):
    """Build a formal preliminary anti-collision PDF report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=16*mm, leftMargin=16*mm,
        topMargin=28*mm, bottomMargin=18*mm,
        title="Preliminary Anti-Collision Assessment",
        author=project.get("prepared_by", "Rigsis") or "Rigsis",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontSize=22, leading=27, alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(name="ReportSub", parent=styles["Normal"], fontSize=11, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#555555")))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=15, leading=19, spaceBefore=4, spaceAfter=8, textColor=colors.HexColor("#1F1F1F")))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=11, leading=14, spaceBefore=6, spaceAfter=5, textColor=colors.HexColor("#333333")))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9.5, leading=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.5, leading=10))
    styles.add(ParagraphStyle(name="TableHeaderWrap", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=5.5, leading=6.5, textColor=colors.white, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="TableCellWrap", parent=styles["Normal"], fontName="Helvetica", fontSize=5.6, leading=6.7, alignment=TA_CENTER))


    story = []
    generated = datetime.now().strftime("%d %B %Y")
    new_well_name = str(project.get("new_well") or "New Well")
    client = project.get("client") or "—"
    field = project.get("field") or "—"
    location = project.get("location") or "—"
    prepared = project.get("prepared_by") or "Rigsis"
    report_no = project.get("report_number") or "—"
    revision = project.get("revision") or "0"

    critical = phase3_summary.iloc[0]
    offset_wells = ", ".join(dict.fromkeys(phase3_summary["Offset Well"].astype(str).tolist()))

    # Cover: deliberately no header/footer.
    story.append(Spacer(1, 25*mm))
    if REPORT_LOGO and os.path.exists(REPORT_LOGO):
        logo = RLImage(REPORT_LOGO)
        logo.drawWidth = 65*mm
        logo.drawHeight = 65*mm * 915/1531
        story.append(logo)
        story.append(Spacer(1, 12*mm))

    # Use the same 512x512 transparent favicon as a small cover identity mark.
    if FAVICON_PATH and os.path.exists(FAVICON_PATH):
        cover_icon = RLImage(FAVICON_PATH)
        cover_icon.drawWidth = 18*mm
        cover_icon.drawHeight = 18*mm
        story.append(cover_icon)
        story.append(Spacer(1, 10*mm))

    story.append(Paragraph("PRELIMINARY", styles["ReportTitle"]))
    story.append(Paragraph("ANTI-COLLISION ASSESSMENT", styles["ReportTitle"]))
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph(f"<b>New Well:</b> {new_well_name}", styles["ReportSub"]))
    story.append(Paragraph(f"<b>Project / Field:</b> {field}", styles["ReportSub"]))
    story.append(Paragraph(f"<b>Location:</b> {location}", styles["ReportSub"]))
    story.append(Spacer(1, 25*mm))
    story.append(Paragraph(f"Prepared by: {prepared}", styles["ReportSub"]))
    story.append(Paragraph(f"Report Date: {generated}", styles["ReportSub"]))
    story.append(Paragraph(f"Report No.: {report_no} &nbsp;&nbsp; Revision: {revision}", styles["ReportSub"]))
    story.append(PageBreak())

    # 1. Report Summary — project identity and engineering result in one non-redundant table.
    story.append(Paragraph("1. Report Summary", styles["H1x"]))
    status_summary = _normalize_screening_status(critical["Screening Status"])
    summary_rows = [
        ["Project / Analysis Information", "Value"],
        ["Project / Field Name", field],
        ["Client / Company", client],
        ["Location", location],
        ["Prepared By", prepared],
        ["Report Number", report_no],
        ["Revision", revision],
        ["Assessment Type", "Preliminary Anti-Collision Screening"],
        ["Report Date", generated],
        ["New Well", new_well_name],
        ["Offset Wells", offset_wells],
        ["Critical Offset Well", str(critical["Offset Well"])],
        ["New Well MD", f"{_fmt(critical['New Well MD (m)'])} m"],
        ["Offset Well MD", f"{_fmt(critical['Offset Well MD (m)'])} m"],
        ["Minimum CtC", f"{_fmt(critical['Center Distance (m)'])} m"],
        ["Minimum SF", _fmt(critical["Separation Factor"])],
        ["Screening Status", Paragraph(status_summary, styles["TableCellWrap"])],
    ]
    story.append(_pdf_table(summary_rows, widths=[60*mm, 100*mm], font_size=8.0, status_col=1))
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph(
        f"The preliminary anti-collision screening identified <b>{critical['Offset Well']}</b> as the critical offset well, "
        f"with a minimum center-to-center distance of <b>{_fmt(critical['Center Distance (m)'])} m</b> at approximately "
        f"<b>{_fmt(critical['TVDSS (m)'])} m TVDSS</b>. The corresponding minimum separation factor is "
        f"<b>{_fmt(critical['Separation Factor'])}</b>, classified as <b>{status_summary}</b> under the assumed positional uncertainty model.",
        styles["Bodyx"]
    ))
    story.append(Paragraph(
        "This document is a preliminary engineering screening and does not replace the detailed anti-collision analysis and operational acceptance performed by the directional drilling service provider.",
        styles["Bodyx"]
    ))

    # 2. New Well Trajectory
    story.append(PageBreak())
    story.append(Paragraph("2. New Well Trajectory", styles["H1x"]))
    if not p3_detailed.empty:
        last = p3_detailed.iloc[-1]
        traj = [
            ["Parameter", "Value"],
            ["Surface Northing", f"{_fmt(p3_detailed['Northing'].iloc[0])} m"],
            ["Surface Easting", f"{_fmt(p3_detailed['Easting'].iloc[0])} m"],
            ["Ground Level", f"{_fmt(project.get('ground_level', '—'))} mASL"],
            ["Rig Elevation", f"{_fmt(project.get('rig_elevation', '—'))} m"],
            ["RKB Elevation", f"{_fmt(project.get('rkb_elevation', '—'))} mASL"],
            ["Target Northing", f"{_fmt(project.get('target_n', '—'))} m"],
            ["Target Easting", f"{_fmt(project.get('target_e', '—'))} m"],
            ["Target Depth / TVDSS", f"{_fmt(project.get('target_depth', '—'))} mASL"],
            ["Final MD", f"{_fmt(last['MD'])} m"],
            ["Final TVD", f"{_fmt(last['TVD'])} m"],
            ["Total Displacement", f"{_fmt(last['Displacement'])} m"],
            ["Final Inclination", f"{_fmt(last['Inclination'])}°"],
            ["Final Azimuth", f"{_fmt(last['Azimuth'])}°"],
        ]
        story.append(_pdf_table(traj, widths=[65*mm, 85*mm], font_size=8))
        story.append(Spacer(1, 5*mm))
        story.append(_figure_to_rl(_plot_new_well(p3_detailed), width_mm=115))

    # 3. New Well vs Offset Wells
    story.append(PageBreak())
    story.append(Paragraph("3. New Well vs Offset Wells", styles["H1x"]))
    if not result.empty and not p3_detailed.empty:
        surface_e = float(p3_detailed["Easting"].iloc[0])
        surface_n = float(p3_detailed["Northing"].iloc[0])
        target_e = float(project.get("target_e", p3_detailed["Easting"].iloc[-1]))
        target_n = float(project.get("target_n", p3_detailed["Northing"].iloc[-1]))
        cmap = build_well_color_map(result)
        story.append(_figure_to_rl(_plot_plan_new_vs_offset(result, p3_detailed, cmap, surface_e, surface_n, target_e, target_n), width_mm=165))

    # 4. Positional Uncertainty Assumption
    story.append(PageBreak())
    story.append(Paragraph("4. Positional Uncertainty Assumption", styles["H1x"]))
    story.append(Paragraph(
        "The screening uses a linear positional uncertainty rate expressed in metres per 1,000 m measured depth. The input values are treated as positional dimensions for the error ellipse and are not angular survey errors.",
        styles["Bodyx"]
    ))
    unc = [
        ["Well Type", "Azimuth Error Rate", "Inclination Error Rate"],
        ["Offset Wells", f"{_fmt(project.get('offset_rate_azi', 0))} m / 1000 m MD", f"{_fmt(project.get('offset_rate_inc', 0))} m / 1000 m MD"],
        ["New Well", f"{_fmt(project.get('new_rate_azi', 0))} m / 1000 m MD", f"{_fmt(project.get('new_rate_inc', 0))} m / 1000 m MD"],
    ]
    story.append(_pdf_table(unc, widths=[55*mm, 55*mm, 55*mm], font_size=8.5))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        "The uncertainty envelope is evaluated on common absolute TVDSS planes at 5 m intervals, beginning 25 m below the highest well surface elevation. Each well contributes only where it physically exists at the common plane.",
        styles["Bodyx"]
    ))

    # 5. Methodology
    story.append(Paragraph("5. Anti-Collision Methodology", styles["H1x"]))
    steps = [
        "Generate common absolute TVDSS check planes at 5 m intervals.",
        "Interpolate the New Well and each Offset Well onto the same check planes.",
        "Calculate positional uncertainty from the specified error rate and MD.",
        "Construct the local positional-error ellipse normal to the local wellbore tangent.",
        "Calculate center-to-center distance (CtC) at each common plane.",
        "Project the uncertainty ellipse in the direction of the other well and calculate the combined error radius.",
        "Calculate Separation Factor as CtC divided by the combined directional uncertainty radius.",
        "Identify the minimum SF for each offset well and the overall critical case.",
    ]
    for i, step in enumerate(steps, 1):
        story.append(Paragraph(f"{i}. {step}", styles["Bodyx"]))
    story.append(Paragraph(
        "Screening interpretation used by the prototype: SF < 1.0 = ENVELOPE OVERLAP - MODIFY TRAJECTORY; 1.0 ≤ SF < 1.5 = CRITICAL - CONSIDER TO MODIFY TRAJECTORY; 1.5 ≤ SF < 3.0 = CLOSE - REVIEW TRAJECTORY; SF ≥ 3.0 = PRELIMINARY PASS. These thresholds are screening indicators only.",
        styles["Bodyx"]
    ))

    # 6. Preliminary Anti-Collision Results
    story.append(PageBreak())
    story.append(Paragraph("6. Preliminary Anti-Collision Results", styles["H1x"]))
    if not uncertainty_result.empty:
        cmap = build_well_color_map(uncertainty_result[uncertainty_result["Well"].astype(str).str.strip().str.lower() != "new well"])
        new_ref = uncertainty_result[uncertainty_result["Well"].astype(str).str.strip().str.lower() == "new well"]
        surface_e = float(new_ref.iloc[0]["X"]); surface_n = float(new_ref.iloc[0]["Y"]); surface_z = float(new_ref.iloc[0]["Z"])
        crit_well = str(critical["Offset Well"])
        crit_z = float(critical["TVDSS (m)"])

        def _interp_xyz(df, z_value):
            work = df.sort_values("Z")
            z = work["Z"].astype(float).to_numpy()
            if len(z) == 0:
                return (np.nan, np.nan, np.nan)
            x = float(np.interp(z_value, z, work["X"].astype(float).to_numpy()))
            y = float(np.interp(z_value, z, work["Y"].astype(float).to_numpy()))
            return x, y, float(z_value)

        new_x, new_y, new_z = _interp_xyz(new_ref, crit_z)
        off_ref = uncertainty_result[uncertainty_result["Well"].astype(str) == crit_well]
        off_x, off_y, off_z = _interp_xyz(off_ref, crit_z)
        critical_full = critical.to_dict()
        critical_full.update({"New X": new_x, "Offset X": off_x, "New Y": new_y, "Offset Y": off_y, "New Z": new_z, "Offset Z": off_z})
        if all(np.isfinite(float(critical_full.get(k, np.nan))) for k in ["New X","Offset X","New Y","Offset Y","New Z","Offset Z"]):
            # 6.1 — trajectory-only 3D view remains the original Matplotlib figure.
            # This does not require Kaleido.
            trajectory_only = _plot_static_3d(
                uncertainty_result, critical_full, cmap,
                surface_e, surface_n, surface_z
            )
            story.append(_figure_to_rl(trajectory_only, width_mm=165))
            story.append(Spacer(1, 4*mm))

            # 6.2 — error-envelope 3D view uses the Phase 3 Plotly figure
            # and Kaleido for PNG export to the PDF.
            report_3d = _plot_plotly_3d_anti_collision(
                uncertainty_result, critical_full, crit_well, cmap,
                surface_e, surface_n, surface_z
            )
            story.append(_plotly_figure_to_rl(report_3d, width_mm=165))

    # 7. CtC & SF Results
    story.append(PageBreak())
    story.append(Paragraph("7. CtC & SF Results", styles["H1x"]))
    cmap = build_well_color_map(result)
    chart_img = _figure_to_rl(_plot_ctc_sf_combined(phase3_details, cmap), width_mm=172)
    story.append(chart_img)

    # 8. Detailed Screening Results
    story.append(PageBreak())
    story.append(Paragraph("8. Detailed Screening Results", styles["H1x"]))
    detail_cols = [
        "Offset Well", "TVDSS (m)", "New Well MD (m)", "Offset Well MD (m)", "Center Distance (m)",
        "New Well Error Radius (m)", "Offset Well Error Radius (m)", "Separation Factor", "Screening Status"
    ]
    d = phase3_summary[detail_cols].copy()
    header_map = {
        "Offset Well": "Offset<br/>Well",
        "TVDSS (m)": "TVDSS<br/>(m)",
        "New Well MD (m)": "New Well MD<br/>(m)",
        "Offset Well MD (m)": "Offset Well MD<br/>(m)",
        "Center Distance (m)": "Center-to-Center<br/>Distance (m)",
        "New Well Error Radius (m)": "New Well<br/>Error Radius (m)",
        "Offset Well Error Radius (m)": "Offset Well<br/>Error Radius (m)",
        "Separation Factor": "Separation<br/>Factor",
        "Screening Status": "Screening<br/>Status",
    }
    data = [[Paragraph(header_map[c], styles["TableHeaderWrap"]) for c in detail_cols]]
    for _, r in d.iterrows():
        status = _normalize_screening_status(r["Screening Status"])
        data.append([
            str(r["Offset Well"]), _fmt(r["TVDSS (m)"]), _fmt(r["New Well MD (m)"]), _fmt(r["Offset Well MD (m)"]),
            _fmt(r["Center Distance (m)"]), _fmt(r["New Well Error Radius (m)"]), _fmt(r["Offset Well Error Radius (m)"]),
            _fmt(r["Separation Factor"]), Paragraph(status, styles["TableCellWrap"])
        ])
    story.append(_pdf_table(data, widths=[23*mm, 14*mm, 17*mm, 18*mm, 17*mm, 19*mm, 19*mm, 13*mm, 25*mm], font_size=5.8, status_col=8))

    # 9. Critical Separation Location & Conclusion
    story.append(Paragraph("9. Critical Separation Location & Conclusion", styles["H1x"]))
    status_display = _normalize_screening_status(critical["Screening Status"])
    story.append(_pdf_table([
        ["Parameter", "Value"],
        ["Critical Offset Well", str(critical["Offset Well"])],
        ["TVDSS", f"{_fmt(critical['TVDSS (m)'])} m"],
        ["New Well MD", f"{_fmt(critical['New Well MD (m)'])} m"],
        ["Offset Well MD", f"{_fmt(critical['Offset Well MD (m)'])} m"],
        ["Center-to-Center Distance", f"{_fmt(critical['Center Distance (m)'])} m"],
        ["Minimum Separation Factor", _fmt(critical["Separation Factor"])],
        ["Screening Status", status_display],
    ], widths=[65*mm, 85*mm], font_size=8.5, status_col=1))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        f"The preliminary anti-collision screening identifies <b>{critical['Offset Well']}</b> as the critical offset well. The minimum CtC is <b>{_fmt(critical['Center Distance (m)'])} m</b> at approximately <b>{_fmt(critical['TVDSS (m)'])} m TVDSS</b>, with a minimum SF of <b>{_fmt(critical['Separation Factor'])}</b> under the stated positional uncertainty assumptions.",
        styles["Bodyx"]
    ))
    story.append(Paragraph(
        "This result is a preliminary screening indicator only. Final anti-collision verification, survey error modelling and operational acceptance shall be detailed by directional drilling contractor in further drilling phase.",
        styles["Bodyx"]
    ))

    # 10. Attachment — detailed survey data.
    if include_survey_attachment:
        story.append(PageBreak())
        story.append(Paragraph("10. Attachment — Detailed Survey Data", styles["H1x"]))
        story.append(Paragraph(
            "The following survey tables are provided as supporting attachment data for review when detailed survey information is required.",
            styles["Bodyx"]
        ))
        if not p3_detailed.empty:
            story.append(Paragraph("Attachment A — New Well Detailed Survey", styles["H2x"]))
            cols = [c for c in ["MD", "TVD", "TVDSS", "Inclination", "Azimuth", "N+", "E+", "Northing", "Easting", "Displacement", "DLS"] if c in p3_detailed.columns]
            head = [Paragraph(c.replace(" ", "<br/>") if c in ["Northing", "Easting", "Displacement"] else c, styles["TableHeaderWrap"]) for c in cols]
            rows = [head]
            for _, r in p3_detailed[cols].iterrows():
                rows.append([_fmt(r[c]) if isinstance(r[c], (int,float,np.integer,np.floating)) else str(r[c]) for c in cols])
            widths = [15*mm] * len(cols)
            story.append(_pdf_table(rows, widths=widths, font_size=5.0))

        if not result.empty:
            story.append(Spacer(1, 5*mm))
            story.append(Paragraph("Attachment B — Offset Wells Detailed Survey", styles["H2x"]))
            cols = [c for c in ["Well", "MD", "TVD", "TVDSS", "Inclination", "Azimuth", "X", "Y", "Z", "DLS"] if c in result.columns]
            head = [Paragraph(c.replace(" ", "<br/>") if c in ["Inclination", "Azimuth"] else c, styles["TableHeaderWrap"]) for c in cols]
            rows = [head]
            for _, r in result[cols].iterrows():
                rows.append([_fmt(r[c]) if isinstance(r[c], (int,float,np.integer,np.floating)) else str(r[c]) for c in cols])
            widths = [18*mm] + [15*mm] * (len(cols)-1)
            story.append(_pdf_table(rows, widths=widths, font_size=5.0))

    def _draw_header_footer(canvas, page_count):
        if canvas._pageNumber == 1:
            return
        w, h = A4
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(colors.HexColor("#222222"))
        canvas.drawString(16*mm, h - 15*mm, "Preliminary Anti Collision Assessment")
        if REPORT_LOGO and os.path.exists(REPORT_LOGO):
            canvas.drawImage(REPORT_LOGO, w - 33.5*mm, h - 18.5*mm, width=15.75*mm, height=15.75*mm*915/1531, preserveAspectRatio=True, mask="auto")
        canvas.setStrokeColor(colors.HexColor("#B8B8B8")); canvas.setLineWidth(0.5)
        canvas.line(16*mm, h - 22*mm, w - 16*mm, h - 22*mm)
        canvas.setFont("Helvetica", 7.5); canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(16*mm, 9*mm, f"Report Date: {generated}")
        canvas.drawCentredString(w / 2.0, 9*mm, str(field))
        canvas.drawRightString(w - 16*mm, 9*mm, f"Page {canvas._pageNumber} of {page_count}")
        canvas.restoreState()

    from reportlab.pdfgen import canvas as pdfcanvas
    class NumberedCanvas(pdfcanvas.Canvas):
        def __init__(self, *args, **kwargs):
            pdfcanvas.Canvas.__init__(self, *args, **kwargs)
            self._saved_page_states = []
        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()
        def save(self):
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                _draw_header_footer(self, num_pages)
                pdfcanvas.Canvas.showPage(self)
            pdfcanvas.Canvas.save(self)

    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()

# Shared well colors used consistently across trajectory, 3D, CtC, and SF plots.
WELL_PALETTE = [
    "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#d62728",
]

def build_well_color_map(df):
    wells = list(dict.fromkeys(df["Well"].astype(str).tolist()))
    return {well: WELL_PALETTE[i % len(WELL_PALETTE)] for i, well in enumerate(wells)}

st.set_page_config(
    page_title="Preliminary Anti-Collision Tool",
    page_icon=FAVICON_PATH if os.path.exists(FAVICON_PATH) else "🧭",
    layout="wide",
)

# Tool header with the same icon used for the browser favicon.
if os.path.exists(FAVICON_PATH):
    try:
        with open(FAVICON_PATH, "rb") as _f:
            _icon_b64 = base64.b64encode(_f.read()).decode("ascii")
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:14px; margin-top:-8px; margin-bottom:4px;">
                <img src="data:image/png;base64,{_icon_b64}"
                     style="width:54px; height:54px; object-fit:contain; flex:0 0 54px;">
                <div style="font-size:2.55rem; line-height:1.15; font-weight:700; color:#FAFAFA;">
                    Preliminary Anti-Collision Tool
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        st.title("Preliminary Anti-Collision Tool")
else:
    st.title("Preliminary Anti-Collision Tool")

st.caption(
    "Phase 1 — Offset Well Data | Phase 2 — New Well Trajectory | Phase 3 — Anti-Collision Analysis"
)

st.info(
    "This prototype is intended for preliminary assessment only. "
    "It is not a replacement for detailed directional drilling / "
    "anti-collision analysis."
)

def create_offset_well_template():
    """Create a blank Excel template matching the required offset-well input format."""
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        pd.DataFrame(columns=["Well", "X", "Y", "Z"]).to_excel(
            writer, sheet_name="Well_Header", index=False
        )
        pd.DataFrame(columns=["Well", "MD", "Inclination", "Azimuth"]).to_excel(
            writer, sheet_name="Survey", index=False
        )

        # Light formatting so the template is immediately understandable.
        wb = writer.book
        for ws in [wb["Well_Header"], wb["Survey"]]:
            ws.freeze_panes = "A2"
            from openpyxl.styles import Font
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for column_cells in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in column_cells)
                ws.column_dimensions[column_cells[0].column_letter].width = max(12, max_len + 3)

    bio.seek(0)
    return bio.getvalue()


template_data = create_offset_well_template()

col_upload, col_template = st.columns([3, 1])
with col_upload:
    uploaded = st.file_uploader(
        "Upload Offset Well Excel File",
        type=["xlsx", "xls"],
    )
with col_template:
    st.write("")
    st.write("")
    st.download_button(
        label="📥 Download Excel Template",
        data=template_data,
        file_name="Offset_Well_Input_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

if uploaded:
    try:
        header_df = pd.read_excel(uploaded, sheet_name="Well_Header")
        survey_df = pd.read_excel(uploaded, sheet_name="Survey")

        st.subheader("1. Uploaded Data")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Well Header")
            st.dataframe(header_df, use_container_width=True)
        with c2:
            st.write("Survey")
            st.dataframe(survey_df, use_container_width=True)

        if st.button("Calculate Offset Well Trajectories", type="primary"):
            result = calculate_all_wells(header_df, survey_df)
            st.session_state["result"] = result
            st.session_state.pop("uncertainty_result", None)
            st.session_state.pop("phase3_summary", None)
            st.session_state.pop("phase3_details", None)
            st.session_state.pop("p3_summary", None)
            st.session_state.pop("p3_detailed", None)
            st.session_state.pop("p3_info", None)
            st.session_state.pop("p3_inputs", None)

    except Exception as e:
        st.error(f"Input data problem: {e}")

if "result" in st.session_state:
    result = st.session_state["result"]

    st.subheader("2. Calculated Trajectory")
    st.dataframe(
        result[
            ["Well", "MD", "Inclination", "Azimuth",
             "TVD", "X", "Y", "Z", "DLS"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("3. Plan View")
    fig_plan = go.Figure()
    well_color_map = build_well_color_map(result)
    for well, df in result.groupby("Well"):
        fig_plan.add_trace(
            go.Scatter(
                x=df["X"],
                y=df["Y"],
                mode="lines+markers",
                name=well,
                marker=dict(size=2),
                line=dict(width=3, color=well_color_map.get(str(well))),
                hovertemplate=(
                    f"<b>{well}</b><br>"
                    "MD=%{customdata[0]:.1f} m<br>"
                    "TVD=%{customdata[1]:.1f} m<br>"
                    "X=%{x:.1f} m<br>"
                    "Y=%{y:.1f} m<extra></extra>"
                ),
                customdata=df[["MD", "TVD"]].values,
            )
        )
    fig_plan.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        yaxis_scaleanchor="x",
        height=600,
    )
    st.plotly_chart(fig_plan, use_container_width=True)

    st.subheader("4. 3D Well View")
    fig3d = go.Figure()
    for well, df in result.groupby("Well"):
        fig3d.add_trace(
            go.Scatter3d(
                x=df["X"],
                y=df["Y"],
                z=df["Z"],
                mode="lines+markers",
                name=well,
                marker=dict(size=2),
                line=dict(width=4, color=well_color_map.get(str(well))),
                hovertemplate=(
                    f"<b>{well}</b><br>"
                    "MD=%{customdata[0]:.1f} m<br>"
                    "TVD=%{customdata[1]:.1f} m<br>"
                    "X=%{x:.1f} m<br>"
                    "Y=%{y:.1f} m<br>"
                    "Z=%{z:.1f} m<extra></extra>"
                ),
                customdata=df[["MD", "TVD"]].values,
            )
        )

    fig3d.update_layout(
        height=700,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z / Elevation",
            aspectmode="data",
        ),
    )
    st.plotly_chart(fig3d, use_container_width=True)

    st.subheader("5. Summary")
    summary = (
        result.groupby("Well")
        .agg(
            Survey_Stations=("MD", "count"),
            MD_Max_m=("MD", "max"),
            TVD_Max_m=("TVD", "max"),
            DLS_Max_deg_30m=("DLS", "max"),
        )
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Calculated Trajectory CSV",
        data=csv,
        file_name="calculated_offset_trajectories.csv",
        mime="text/csv",
    )


# ======================================================================
# PHASE 2 — NEW WELL TRAJECTORY
# ======================================================================

st.divider()
with st.expander("Phase 2 — New Well Trajectory", expanded=False):
    st.subheader("Phase 2 — New Well Trajectory")
    st.write(
        "Generate a preliminary J-Type trajectory for the proposed new well. "
        "The trajectory uses constant azimuth and follows the same engineering "
        "coordinate convention used by the prototype: X/Northing, Y/Easting, "
        "and Z/elevation."
    )

    def p3_calculate_dogleg_angle(inc1, inc2, azi1, azi2):
        i1 = radians(inc1)
        i2 = radians(inc2)
        da = radians(azi2 - azi1)
        value = (
            cos(i1) * cos(i2)
            + sin(i1) * sin(i2) * cos(da)
        )
        value = max(-1.0, min(1.0, value))
        return degrees(acos(value))


    def p3_calculate_bur_from_kop_inc(target_tvd, hd_target, kop, target_inc):
        I = radians(target_inc)
        den = (1 - cos(I)) - sin(I) * tan(I)
        num = hd_target - (target_tvd - kop) * tan(I)

        if abs(den) < 1e-12:
            raise ValueError("Geometri tidak stabil untuk inclination tersebut.")

        R = num / den
        if R <= 0:
            raise ValueError(
                "Kombinasi KOP, target dan inclination tidak menghasilkan BUR positif."
            )

        return degrees(1 / R) * 30


    def p3_calculate_inc_from_kop_bur(target_tvd, hd_target, kop, bur):
        R = 30 / radians(bur)

        def f(I):
            return (
                R * (1 - cos(I))
                + (target_tvd - kop - R * sin(I)) * tan(I)
                - hd_target
            )

        grid = np.linspace(radians(0.1), radians(89.0), 1000)
        vals = [f(x) for x in grid]

        for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
            if fa == 0:
                return degrees(a)
            if fa * fb < 0:
                for _ in range(100):
                    m = (a + b) / 2
                    fm = f(m)
                    if abs(fm) < 1e-10:
                        break
                    if fa * fm <= 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                return degrees((a + b) / 2)

        raise ValueError(
            "Tidak ditemukan target inclination yang memenuhi kombinasi KOP + BUR."
        )


    def p3_calculate_kop_from_inc_bur(target_tvd, target_inc, bur):
        R = 30 / radians(bur)
        kop = target_tvd - R * sin(radians(target_inc))
        if kop < 0:
            raise ValueError(
                "KOP hasil perhitungan negatif. Kombinasi target, inclination dan BUR tidak valid."
            )
        return kop


    def p3_build_trajectory(
        surface_n, surface_e, rkb_elev,
        target_n, target_e, target_depth_masl,
        kop, bur, target_inc,
        td_mode="Automatic",
        custom_td=None,
        station_interval=30.0,
    ):
        delta_n = target_n - surface_n
        delta_e = target_e - surface_e
        hd_target = sqrt(delta_n**2 + delta_e**2)
        target_tvd = rkb_elev - target_depth_masl
        azimuth = degrees(atan2(delta_e, delta_n)) % 360.0

        R = 30.0 / radians(bur)
        I = radians(target_inc)

        build_md = R * I
        eob = kop + build_md
        build_tvd = R * sin(I)
        build_hd = R * (1 - cos(I))

        tangent_tvd = target_tvd - (kop + build_tvd)
        tangent_length = tangent_tvd / cos(I)

        if tangent_length < -1e-8:
            raise ValueError(
                "Target terlalu dangkal untuk kombinasi KOP/BUR/inclination."
            )

        calculated_hd = build_hd + max(0.0, tangent_length) * sin(I)
        if abs(calculated_hd - hd_target) > max(0.5, hd_target * 0.005):
            raise ValueError(
                f"Trajectory geometry tidak menutup ke target. "
                f"Calculated HD={calculated_hd:.2f} m, "
                f"target HD={hd_target:.2f} m."
            )

        target_md = eob + max(0.0, tangent_length)

        if td_mode == "Automatic":
            td = math.ceil(target_md / station_interval - 1e-12) * station_interval
        else:
            td = float(custom_td)
            if td < target_md - 1e-8:
                raise ValueError(
                    f"Custom TD harus >= Target MD ({target_md:.2f} m)."
                )

        regular = list(np.arange(0, td + station_interval * 0.01, station_interval))
        key_points = [0.0, kop, eob, target_md, td]
        mds = sorted(
            set(
                round(float(x), 6)
                for x in regular + key_points
                if 0 <= x <= td + 1e-8
            )
        )

        rows = []

        for md in mds:
            if md <= kop + 1e-9:
                tvd = md
                inc = 0.0
                n_plus = 0.0
                e_plus = 0.0
                parameter = "Vertical"

            elif md <= eob + 1e-9:
                build_md_local = md - kop
                theta = build_md_local / R
                inc = degrees(theta)
                tvd = kop + R * sin(theta)
                hd_local = R * (1 - cos(theta))
                n_plus = hd_local * cos(radians(azimuth))
                e_plus = hd_local * sin(radians(azimuth))
                parameter = "Build"

            else:
                inc = target_inc
                tangent_md_local = md - eob
                tvd = (
                    kop
                    + build_tvd
                    + tangent_md_local * cos(I)
                )
                hd_local = (
                    build_hd
                    + tangent_md_local * sin(I)
                )
                n_plus = hd_local * cos(radians(azimuth))
                e_plus = hd_local * sin(radians(azimuth))
                parameter = "Tangent"

            northing = surface_n + n_plus
            easting = surface_e + e_plus
            z_masl = rkb_elev - tvd

            if rows:
                prev = rows[-1]
                delta_md = md - prev["MD"]
                dls = (
                    p3_calculate_dogleg_angle(
                        prev["Inclination"],
                        inc,
                        prev["Azimuth"],
                        azimuth if inc > 0 else 0.0,
                    ) / delta_md * 30
                    if delta_md > 0 else 0.0
                )
            else:
                dls = 0.0

            rows.append({
                "MD": md,
                "TVD": tvd,
                "Inclination": inc,
                "Azimuth": azimuth if md > kop else 0.0,
                "N+": n_plus,
                "E+": e_plus,
                "Northing": northing,
                "Easting": easting,
                "Displacement": sqrt(n_plus**2 + e_plus**2),
                "TVDSS": z_masl,
                "BUR": dls,
                "Parameter": parameter,
            })

        detailed = pd.DataFrame(rows)

        target_rows = detailed.iloc[
            (detailed["MD"] - target_md).abs().argsort()[:1]
        ]
        target_idx = target_rows.index[0]

        detailed.loc[target_idx, "Northing"] = target_n
        detailed.loc[target_idx, "Easting"] = target_e
        detailed.loc[target_idx, "N+"] = delta_n
        detailed.loc[target_idx, "E+"] = delta_e
        detailed.loc[target_idx, "Displacement"] = hd_target
        detailed.loc[target_idx, "TVDSS"] = target_depth_masl
        detailed.loc[target_idx, "TVD"] = target_tvd
        detailed.loc[target_idx, "Inclination"] = target_inc
        detailed.loc[target_idx, "Azimuth"] = azimuth
        detailed.loc[target_idx, "Parameter"] = "Target"

        # Summary key points.
        eob_n = build_hd * cos(radians(azimuth))
        eob_e = build_hd * sin(radians(azimuth))

        td_row = detailed.iloc[-1]

        summary = pd.DataFrame([
            [
                "RKB", 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, surface_n, surface_e,
                0.0, rkb_elev, 0.0
            ],
            [
                "KOP", kop, kop, 0.0, 0.0,
                0.0, 0.0, surface_n, surface_e,
                0.0, rkb_elev - kop, 0.0
            ],
            [
                "EOB", eob, kop + build_tvd, target_inc, azimuth,
                eob_n, eob_e,
                surface_n + eob_n,
                surface_e + eob_e,
                build_hd,
                rkb_elev - kop - build_tvd,
                bur
            ],
            [
                "Target", target_md, target_tvd, target_inc, azimuth,
                delta_n, delta_e, target_n, target_e,
                hd_target, target_depth_masl, 0.0
            ],
            [
                "TD", td,
                float(td_row["TVD"]),
                target_inc,
                azimuth,
                float(td_row["N+"]),
                float(td_row["E+"]),
                float(td_row["Northing"]),
                float(td_row["Easting"]),
                float(td_row["Displacement"]),
                float(td_row["TVDSS"]),
                0.0
            ],
        ], columns=[
            "Parameter", "MD", "TVD", "Inc", "Azimuth",
            "N+", "E+", "Northing", "Easting",
            "Displacement", "TVDSS", "BUR"
        ])

        info = {
            "hd_target": hd_target,
            "target_tvd": target_tvd,
            "target_azimuth": azimuth,
            "kop": kop,
            "eob": eob,
            "target_md": target_md,
            "td": td,
            "target_inc": target_inc,
            "bur": bur,
            "rkb_elevation": rkb_elev,
            "target_depth_masl": target_depth_masl,
        }

        return summary, detailed, info


    def p3_excel_bytes(summary, detailed, inputs):
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            pd.DataFrame(
                inputs.items(), columns=["Input", "Value"]
            ).to_excel(
                writer, sheet_name="Design Inputs", index=False
            )
            summary.to_excel(
                writer, sheet_name="Summary", index=False
            )
            detailed.to_excel(
                writer, sheet_name="Detailed Survey", index=False
            )
        return bio.getvalue()


    # ----------------------------------------------------------------------
    # INPUTS — keep terminology aligned with the user's previous tool
    # ----------------------------------------------------------------------

    st.subheader("1. Surface Point")

    p3_c1, p3_c2, p3_c3, p3_c4 = st.columns(4)

    with p3_c1:
        p3_surface_n = st.number_input(
            "Northing (m)",
            value=9203850.00,
            format="%.2f",
            key="p3_surface_n",
        )

    with p3_c2:
        p3_surface_e = st.number_input(
            "Easting (m)",
            value=376917.00,
            format="%.2f",
            key="p3_surface_e",
        )

    with p3_c3:
        p3_ground = st.number_input(
            "Ground Level (mASL)",
            value=1923.00,
            format="%.2f",
            key="p3_ground",
        )

    with p3_c4:
        p3_rig = st.number_input(
            "Rig Elevation (m)",
            value=0.00,
            format="%.2f",
            key="p3_rig",
        )

    p3_rkb = p3_ground + p3_rig
    st.write(f"**RKB Elevation, mASL:** {p3_rkb:,.2f}")

    st.subheader("2. Subsurface Target")

    p3_t1, p3_t2, p3_t3 = st.columns(3)

    with p3_t1:
        p3_target_n = st.number_input(
            "Target Northing (m)",
            value=9204720.02,
            format="%.2f",
            key="p3_target_n",
        )

    with p3_t2:
        p3_target_e = st.number_input(
            "Target Easting (m)",
            value=377526.19,
            format="%.2f",
            key="p3_target_e",
        )

    with p3_t3:
        p3_target_z = st.number_input(
            "Target Depth (mASL)",
            value=-801.00,
            format="%.2f",
            key="p3_target_z",
        )

    p3_delta_n = p3_target_n - p3_surface_n
    p3_delta_e = p3_target_e - p3_surface_e
    p3_hd = sqrt(p3_delta_n**2 + p3_delta_e**2)
    p3_target_tvd = p3_rkb - p3_target_z
    p3_target_azimuth = degrees(
        atan2(p3_delta_e, p3_delta_n)
    ) % 360.0

    st.subheader("3. Design Parameters")

    p3_method = st.radio(
        "Calculation Method",
        [
            "KOP + Inclination",
            "KOP + BUR",
            "Inclination + BUR",
        ],
        key="p3_method",
    )

    p3_d1, p3_d2, p3_d3 = st.columns(3)

    if p3_method == "KOP + Inclination":

        with p3_d1:
            p3_kop = st.number_input(
                "KOP (m)",
                min_value=0.0,
                value=700.0,
                step=10.0,
                format="%.2f",
                key="p3_kop_1",
            )

        with p3_d2:
            p3_inc = st.number_input(
                "Target Inclination (deg)",
                min_value=0.1,
                max_value=89.0,
                value=30.0,
                step=1.0,
                format="%.2f",
                key="p3_inc_1",
            )

        try:
            p3_bur = p3_calculate_bur_from_kop_inc(
                p3_target_tvd, p3_hd, p3_kop, p3_inc
            )
            with p3_d3:
                st.metric("Calculated BUR", f"{p3_bur:.3f} deg/30m")
        except Exception as exc:
            p3_bur = None
            st.error(str(exc))

    elif p3_method == "KOP + BUR":

        with p3_d1:
            p3_kop = st.number_input(
                "KOP (m)",
                min_value=0.0,
                value=700.0,
                step=10.0,
                format="%.2f",
                key="p3_kop_2",
            )

        with p3_d2:
            p3_bur = st.number_input(
                "BUR (deg/30m)",
                min_value=0.01,
                value=2.0,
                step=0.1,
                format="%.2f",
                key="p3_bur_2",
            )

        try:
            p3_inc = p3_calculate_inc_from_kop_bur(
                p3_target_tvd, p3_hd, p3_kop, p3_bur
            )
            with p3_d3:
                st.metric(
                    "Calculated Target Inclination",
                    f"{p3_inc:.3f} deg"
                )
        except Exception as exc:
            p3_inc = None
            st.error(str(exc))

    else:

        with p3_d1:
            p3_inc = st.number_input(
                "Target Inclination (deg)",
                min_value=0.1,
                max_value=89.0,
                value=30.0,
                step=1.0,
                format="%.2f",
                key="p3_inc_3",
            )

        with p3_d2:
            p3_bur = st.number_input(
                "BUR (deg/30m)",
                min_value=0.01,
                value=2.0,
                step=0.1,
                format="%.2f",
                key="p3_bur_3",
            )

        try:
            p3_kop = p3_calculate_kop_from_inc_bur(
                p3_target_tvd, p3_inc, p3_bur
            )
            with p3_d3:
                st.metric("Calculated KOP", f"{p3_kop:.1f} m")
        except Exception as exc:
            p3_kop = None
            st.error(str(exc))

    st.subheader("4. Total Depth (TD) Settings")

    p3_td_mode = st.radio(
        "TD Mode",
        [
            "Automatic Standdown Depth (Nearest 30m Multiple after Target Depth)",
            "Custom Depth",
        ],
        key="p3_td_mode",
    )

    p3_custom_td = None
    if p3_td_mode == "Custom Depth":
        p3_custom_td = st.number_input(
            "Custom TD (m MD)",
            min_value=0.0,
            value=3030.0,
            step=30.0,
            format="%.2f",
            key="p3_custom_td",
        )

    if st.button(
        "Calculate New Well Trajectory",
        type="primary",
        use_container_width=True,
        key="p3_calculate",
    ):
        try:
            if p3_kop is None or p3_bur is None or p3_inc is None:
                raise ValueError("Parameter trajectory belum valid.")

            p3_td_mode_calc = (
                "Automatic"
                if p3_td_mode.startswith("Automatic")
                else "Custom"
            )

            p3_summary, p3_detailed, p3_info = p3_build_trajectory(
                p3_surface_n,
                p3_surface_e,
                p3_rkb,
                p3_target_n,
                p3_target_e,
                p3_target_z,
                float(p3_kop),
                float(p3_bur),
                float(p3_inc),
                p3_td_mode_calc,
                p3_custom_td,
                30.0,
            )

            p3_inputs = {
                "Surface Northing (m)": p3_surface_n,
                "Surface Easting (m)": p3_surface_e,
                "Ground Level (mASL)": p3_ground,
                "Rig Elevation (m)": p3_rig,
                "RKB Elevation (mASL)": p3_rkb,
                "Target Northing (m)": p3_target_n,
                "Target Easting (m)": p3_target_e,
                "Target Depth (mASL)": p3_target_z,
                "Calculation Method": p3_method,
                "KOP (m)": p3_kop,
                "BUR (deg/30m)": p3_bur,
                "Target Inclination (deg)": p3_inc,
                "TD Mode": p3_td_mode,
                "Custom TD (m)": (
                    p3_custom_td if p3_custom_td is not None else ""
                ),
            }

            st.session_state["p3_summary"] = p3_summary
            st.session_state["p3_detailed"] = p3_detailed
            st.session_state["p3_info"] = p3_info
            st.session_state["p3_inputs"] = p3_inputs
            st.session_state.pop("uncertainty_result", None)
            st.session_state.pop("phase4_summary", None)
            st.session_state.pop("phase4_details", None)

        except Exception as exc:
            st.error(f"Trajectory tidak dapat dibuat: {exc}")

    if "p3_summary" in st.session_state:

        p3_summary = st.session_state["p3_summary"]
        p3_detailed = st.session_state["p3_detailed"]
        p3_info = st.session_state["p3_info"]
        p3_inputs = st.session_state["p3_inputs"]

        st.header("Trajectory Results Summary")

        p3_target_row = p3_detailed[
            p3_detailed["Parameter"] == "Target"
        ]

        if not p3_target_row.empty:
            tr = p3_target_row.iloc[0]
            p3_remaining = sqrt(
                (p3_target_n - tr["Northing"])**2
                + (p3_target_e - tr["Easting"])**2
            )
        else:
            p3_remaining = float("nan")

        st.markdown(
            f"**Distance to target:** "
            f"<span style='color:#00c853'>{p3_remaining:.2f} m</span>",
            unsafe_allow_html=True,
        )

        p3_excel = p3_excel_bytes(
            p3_summary, p3_detailed, p3_inputs
        )

        p3_b1, p3_b2 = st.columns(2)

        with p3_b1:
            st.download_button(
                "📥 Download Excel Sheet (Summary)",
                data=p3_excel,
                file_name="new_well_trajectory_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="p3_download_summary",
            )

        with p3_b2:
            st.download_button(
                "📥 Download Excel Sheet (Detailed)",
                data=p3_excel,
                file_name="new_well_trajectory_detailed.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="p3_download_detailed",
            )

        p3_display_summary = p3_summary.copy()
        for col in [
            "MD", "TVD", "Inc", "Azimuth", "N+", "E+",
            "Northing", "Easting", "Displacement", "TVDSS", "BUR"
        ]:
            p3_display_summary[col] = p3_display_summary[col].map(
                lambda x: f"{x:,.2f}"
            )

        st.dataframe(
            p3_display_summary,
            use_container_width=True,
            hide_index=True,
        )

        st.header("Trajectory Visualization")

        p3_vcol, p3_acol = st.columns(2)

        # Static image-style Vertical View.
        with p3_vcol:
            fig_v, ax_v = plt.subplots(figsize=(6.2, 10.0))

            ax_v.plot(
                p3_detailed["Displacement"],
                p3_detailed["TVD"],
                color="blue",
                linewidth=2.5,
                label="New Well",
            )
            ax_v.scatter(
                [p3_hd],
                [p3_target_tvd],
                color="red",
                marker="x",
                s=95,
                linewidths=2.2,
                label="Target",
                zorder=5,
            )

            max_disp = max(
                float(p3_detailed["Displacement"].max()),
                float(p3_hd),
                1.0,
            )
            max_tvd = max(
                float(p3_detailed["TVD"].max()),
                float(p3_target_tvd),
                1.0,
            )

            x_step = 250.0
            y_step = 500.0

            x_max = math.ceil(max_disp / x_step) * x_step
            y_max = math.ceil(max_tvd / y_step) * y_step

            if x_max <= max_disp + 1e-9:
                x_max += x_step
            if y_max <= max_tvd + 1e-9:
                y_max += y_step

            # Keep a 50 m left-side margin so the vertical section does not sit directly on the Y-axis.
            ax_v.set_xlim(-50, x_max)
            ax_v.set_ylim(y_max * 1.04, 0)
            ax_v.set_xticks(np.arange(0, x_max + 0.1, x_step))
            ax_v.set_yticks(np.arange(0, y_max + 0.1, y_step))

            ax_v.set_title(
                "VERTICAL VIEW",
                fontweight="bold",
                fontsize=13,
            )
            ax_v.set_xlabel("Displacement (m)", fontsize=10)
            ax_v.set_ylabel("TVD (m)", fontsize=10)
            ax_v.grid(
                True,
                linestyle=":",
                linewidth=1.0,
                alpha=0.65,
            )
            ax_v.legend(loc="upper right", fontsize=9)

            fig_v.tight_layout()
            st.pyplot(fig_v, use_container_width=True)
            plt.close(fig_v)

        # Static image-style Azimuth View.
        with p3_acol:
            fig_a, ax_a = plt.subplots(figsize=(6.2, 6.2))

            ax_a.plot(
                p3_detailed["E+"],
                p3_detailed["N+"],
                color="blue",
                linewidth=2.5,
                label="Trajectory",
            )
            ax_a.scatter(
                [p3_delta_e],
                [p3_delta_n],
                color="red",
                marker="x",
                s=95,
                linewidths=2.2,
                label="Target",
                zorder=5,
            )
            ax_a.scatter(
                [0],
                [0],
                color="black",
                marker="o",
                s=55,
                label="Surface (0,0)",
                zorder=5,
            )

            max_xy = max(
                float(np.max(np.abs(p3_detailed["E+"]))),
                float(np.max(np.abs(p3_detailed["N+"]))),
                abs(float(p3_delta_e)),
                abs(float(p3_delta_n)),
                1.0,
            )

            xy_step = 200.0
            xy_limit = math.ceil(max_xy / xy_step) * xy_step

            if xy_limit <= max_xy + 1e-9:
                xy_limit += xy_step

            ax_a.set_xlim(-xy_limit, xy_limit)
            ax_a.set_ylim(-xy_limit, xy_limit)
            ax_a.set_xticks(
                np.arange(-xy_limit, xy_limit + 0.1, xy_step)
            )
            ax_a.set_yticks(
                np.arange(-xy_limit, xy_limit + 0.1, xy_step)
            )

            ax_a.set_title(
                "AZIMUTH VIEW",
                fontweight="bold",
                fontsize=13,
            )
            ax_a.set_xlabel("E+", fontsize=10)
            ax_a.set_ylabel("N+", fontsize=10)
            ax_a.grid(
                True,
                linestyle=":",
                linewidth=1.0,
                alpha=0.65,
            )
            ax_a.set_aspect("equal", adjustable="box")
            ax_a.legend(loc="upper left", fontsize=9)

            fig_a.tight_layout()
            st.pyplot(fig_a, use_container_width=True)
            plt.close(fig_a)


        # ------------------------------------------------------------------
        # Combined trajectory view for trial-and-error screening.
        # This intentionally shows trajectories only (no uncertainty envelope).
        # Phase 3 will handle uncertainty / anti-collision calculations.
        # ------------------------------------------------------------------
        if "result" in st.session_state:
            st.header("New Well vs Offset Wells")
            st.caption(
                "Visual screening for trial-and-error design. "
                "Offset wells are shown together with the current New Well trajectory."
            )

            offset_result = st.session_state["result"]
            well_color_map = build_well_color_map(offset_result)

            p3_cv1, p3_cv2 = st.columns(2)

            # Local coordinates are centered on the New Well surface so that
            # the relative position is immediately readable.
            new_surface_e = float(p3_surface_e)
            new_surface_n = float(p3_surface_n)
            new_surface_z = float(p3_rkb)

            with p3_cv1:
                fig_cp, ax_cp = plt.subplots(figsize=(6.2, 6.2))

                for well, odf in offset_result.groupby("Well"):
                    ax_cp.plot(
                        odf["X"] - new_surface_e,
                        odf["Y"] - new_surface_n,
                        linewidth=1.6,
                        label=str(well),
                        color=well_color_map.get(str(well)),
                        alpha=0.75,
                    )

                ax_cp.plot(
                    p3_detailed["E+"],
                    p3_detailed["N+"],
                    linewidth=3.0,
                    label="New Well",
                    zorder=5,
                )
                ax_cp.scatter(
                    [0], [0],
                    marker="o",
                    s=45,
                    label="New Well Surface",
                    zorder=6,
                )
                ax_cp.scatter(
                    [p3_delta_e], [p3_delta_n],
                    marker="x",
                    s=90,
                    linewidths=2.0,
                    label="Target",
                    zorder=7,
                )

                all_x = []
                all_y = []
                for _, odf in offset_result.groupby("Well"):
                    all_x.extend((odf["X"] - new_surface_e).astype(float).tolist())
                    all_y.extend((odf["Y"] - new_surface_n).astype(float).tolist())
                all_x.extend(p3_detailed["E+"].astype(float).tolist())
                all_y.extend(p3_detailed["N+"].astype(float).tolist())
                all_x.extend([float(p3_delta_e), 0.0])
                all_y.extend([float(p3_delta_n), 0.0])

                max_local = max(
                    max(abs(v) for v in all_x) if all_x else 1.0,
                    max(abs(v) for v in all_y) if all_y else 1.0,
                    1.0,
                )
                # Use a readable engineering scale, while preserving equal X/Y scale.
                xy_step = 200.0
                xy_limit = math.ceil(max_local / xy_step) * xy_step
                if xy_limit <= max_local + 1e-9:
                    xy_limit += xy_step

                ax_cp.set_xlim(-xy_limit, xy_limit)
                ax_cp.set_ylim(-xy_limit, xy_limit)
                ax_cp.set_xticks(np.arange(-xy_limit, xy_limit + 0.1, xy_step))
                ax_cp.set_yticks(np.arange(-xy_limit, xy_limit + 0.1, xy_step))
                ax_cp.set_title("PLAN VIEW — NEW WELL vs OFFSET WELLS", fontweight="bold", fontsize=12)
                ax_cp.set_xlabel("ΔEasting from New Well Surface (m)", fontsize=9)
                ax_cp.set_ylabel("ΔNorthing from New Well Surface (m)", fontsize=9)
                ax_cp.grid(True, linestyle=":", linewidth=1.0, alpha=0.65)
                ax_cp.set_aspect("equal", adjustable="box")
                ax_cp.legend(loc="best", fontsize=8)
                fig_cp.tight_layout()
                st.pyplot(fig_cp, use_container_width=True)
                plt.close(fig_cp)

            with p3_cv2:
                # Interactive 3D view using Plotly.
                # Local coordinates are centered on the New Well surface.
                fig_c3 = go.Figure()

                for well, odf in offset_result.groupby("Well"):
                    fig_c3.add_trace(
                        go.Scatter3d(
                            x=(odf["X"] - new_surface_e).astype(float),
                            y=(odf["Y"] - new_surface_n).astype(float),
                            z=(odf["Z"] - new_surface_z).astype(float),
                            mode="lines",
                            name=str(well),
                            line=dict(width=4, color=well_color_map.get(str(well))),
                            hovertemplate=(
                                f"{well}<br>"
                                "ΔEasting: %{x:.2f} m<br>"
                                "ΔNorthing: %{y:.2f} m<br>"
                                "ΔElevation: %{z:.2f} m<extra></extra>"
                            ),
                        )
                    )

                fig_c3.add_trace(
                    go.Scatter3d(
                        x=p3_detailed["E+"].astype(float),
                        y=p3_detailed["N+"].astype(float),
                        z=(p3_detailed["TVDSS"] - new_surface_z).astype(float),
                        mode="lines",
                        name="New Well",
                        line=dict(width=7, color="#ff4b4b"),
                        hovertemplate=(
                            "New Well<br>"
                            "MD: %{customdata[0]:.2f} m<br>"
                            "ΔEasting: %{x:.2f} m<br>"
                            "ΔNorthing: %{y:.2f} m<br>"
                            "ΔElevation: %{z:.2f} m<extra></extra>"
                        ),
                        customdata=p3_detailed[["MD"]].astype(float).to_numpy(),
                    )
                )

                fig_c3.add_trace(
                    go.Scatter3d(
                        x=[0], y=[0], z=[0],
                        mode="markers",
                        name="New Well Surface",
                        marker=dict(size=5, color="#2ecc71"),
                        hovertemplate=(
                            "New Well Surface<br>"
                            "ΔEasting: 0.00 m<br>"
                            "ΔNorthing: 0.00 m<br>"
                            "ΔElevation: 0.00 m<extra></extra>"
                        ),
                    )
                )

                fig_c3.add_trace(
                    go.Scatter3d(
                        x=[float(p3_delta_e)],
                        y=[float(p3_delta_n)],
                        z=[float(p3_target_z - new_surface_z)],
                        mode="markers",
                        name="Target",
                        marker=dict(size=7, symbol="x", color="#19d3c5"),
                        hovertemplate=(
                            "Target<br>"
                            "ΔEasting: %{x:.2f} m<br>"
                            "ΔNorthing: %{y:.2f} m<br>"
                            "ΔElevation: %{z:.2f} m<extra></extra>"
                        ),
                    )
                )

                fig_c3.update_layout(
                    title=dict(
                        text="3D VIEW — NEW WELL vs OFFSET WELLS",
                        x=0,
                        xanchor="left",
                        y=0.99,
                        yanchor="top",
                        font=dict(size=15),
                    ),
                    height=650,
                    margin=dict(l=0, r=0, t=85, b=0),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.08,
                        xanchor="left",
                        x=0.28,
                        font=dict(size=11),
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=0,
                        entrywidth=95,
                        itemsizing="constant",
                    ),
                    scene=dict(
                        xaxis_title="ΔEasting (m)",
                        yaxis_title="ΔNorthing (m)",
                        zaxis_title="ΔElevation (m)",
                        aspectmode="auto",
                        camera=dict(
                            eye=dict(x=1.55, y=1.55, z=1.25)
                        ),
                    ),
                )
                st.plotly_chart(fig_c3, use_container_width=True, config={"displaylogo": False})

        st.header("Detailed Survey Results")

        p3_display_detail = p3_detailed.copy()
        for col in [
            "MD", "TVD", "Inclination", "Azimuth", "N+", "E+",
            "Northing", "Easting", "Displacement", "TVDSS", "BUR"
        ]:
            p3_display_detail[col] = p3_display_detail[col].map(
                lambda x: f"{x:,.2f}"
            )

        st.dataframe(
            p3_display_detail,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Phase 3 is a preliminary trajectory design. "
            "Detailed directional drilling and anti-collision design "
            "remain subject to final verification by the directional "
            "drilling contractor."
        )

# ======================================================================
# ======================================================================
# PHASE 3 — ANTI-COLLISION ANALYSIS
# ======================================================================

if "result" in st.session_state and "p3_detailed" in st.session_state:
    st.divider()
    st.header("Phase 3 — Anti-Collision Analysis")
    st.write(
        "Input the positional uncertainty for Offset Wells and New Well, "
        "then run the preliminary anti-collision analysis. The uncertainty "
        "envelope calculation is performed automatically in the background."
    )

    st.subheader("1. Positional Uncertainty Input")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**OFFSET WELL UNCERTAINTY**")
        st.markdown("**Azimuth Error Rate**")
        offset_rate_azi = st.number_input(
            "Rate (m / 1000 m MD)", min_value=0.0, value=5.0, step=0.5, key="offset_rate_azi"
        )
        st.markdown("**Inclination Error Rate**")
        offset_rate_inc = st.number_input(
            "Rate (m / 1000 m MD)", min_value=0.0, value=3.0, step=0.5, key="offset_rate_inc"
        )
    with col2:
        st.markdown("**NEW WELL UNCERTAINTY**")
        st.markdown("**Azimuth Error Rate**")
        new_rate_azi = st.number_input(
            "Rate (m / 1000 m MD)", min_value=0.0, value=5.0, step=0.5, key="new_rate_azi"
        )
        st.markdown("**Inclination Error Rate**")
        new_rate_inc = st.number_input(
            "Rate (m / 1000 m MD)", min_value=0.0, value=3.0, step=0.5, key="new_rate_inc"
        )

    st.caption(
        "Error rate adalah positional uncertainty dalam meter per 1000 m MD, "
        "bukan angular error. Uncertainty bertambah linear sepanjang MD dari awal well. "
        "Screening menggunakan TVDSS dengan interval tetap 5 m, dimulai 25 m di bawah "
        "elevasi surface tertinggi."
    )

    if st.button(
        "Run Anti-Collision Analysis",
        type="primary",
        use_container_width=True,
        key="phase3_run_anti_collision",
    ):
        try:
            # Build New Well dataframe from the latest calculated trajectory.
            new_df = st.session_state["p3_detailed"].copy()
            new_df["Well"] = "New Well"
            new_df["X"] = new_df["Easting"].astype(float)
            new_df["Y"] = new_df["Northing"].astype(float)
            new_df["Z"] = new_df["TVDSS"].astype(float)
            new_df["Inclination"] = new_df["Inclination"].astype(float)
            new_df["Azimuth"] = new_df["Azimuth"].astype(float)
            new_df["TVD"] = new_df["TVD"].astype(float)
            new_df["DLS"] = new_df.get("BUR", 0.0)

            # Offset wells from Phase 1.
            offset_df = st.session_state["result"].copy()
            combined_df = pd.concat(
                [
                    offset_df,
                    new_df[
                        [
                            "Well", "MD", "Inclination", "Azimuth", "TVD",
                            "X", "Y", "Z", "DLS"
                        ]
                    ],
                ],
                ignore_index=True,
                sort=False,
            )

            error_params = {
                "offset": (offset_rate_azi, offset_rate_inc),
                "new": (new_rate_azi, new_rate_inc),
            }

            # Uncertainty is an internal calculation now; the user does not
            # need to generate a separate envelope phase.
            uncertainty_result = add_depth_dependent_uncertainty(
                combined_df,
                error_params=error_params,
            )

            phase3_summary, phase3_details = preliminary_anti_collision(
                uncertainty_result[
                    uncertainty_result["Well"].astype(str).str.strip().str.lower() == "new well"
                ].copy(),
                uncertainty_result[
                    uncertainty_result["Well"].astype(str).str.strip().str.lower() != "new well"
                ].copy(),
                interval_z=5.0,
                start_offset=25.0,
            )

            if phase3_summary.empty:
                raise ValueError(
                    "Tidak ada pasangan New Well vs Offset Well yang dapat dihitung "
                    "pada rentang TVDSS yang sama."
                )

            st.session_state["uncertainty_result"] = uncertainty_result
            st.session_state["phase3_summary"] = phase3_summary
            st.session_state["phase3_details"] = phase3_details

        except Exception as exc:
            st.error(f"Anti-collision analysis tidak dapat dijalankan: {exc}")

    if "phase3_summary" in st.session_state:
        phase3_summary = st.session_state["phase3_summary"].copy()
        phase3_details = st.session_state.get("phase3_details", pd.DataFrame()).copy()
        uresult = st.session_state.get("uncertainty_result", pd.DataFrame()).copy()

        st.subheader("2. Anti-Collision Summary")
        critical = phase3_summary.iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Critical Offset Well", str(critical["Offset Well"]))
        with m2:
            st.metric("Critical TVDSS", f"{critical['TVDSS (m)']:.2f} m")
        with m3:
            sf_text = (
                f"{critical['Separation Factor']:.2f}"
                if np.isfinite(critical["Separation Factor"])
                else "∞"
            )
            st.metric("Minimum Separation Factor", sf_text)
        with m4:
            st.metric("Screening Status", str(critical["Screening Status"]))

        display_cols = [
            "Offset Well",
            "TVDSS (m)",
            "New Well MD (m)",
            "Offset Well MD (m)",
            "Center Distance (m)",
            "New Well Error Radius (m)",
            "Offset Well Error Radius (m)",
            "Separation Factor",
            "Screening Status",
        ]
        display_summary = phase3_summary[display_cols].copy()
        for col in display_cols[1:-1]:
            display_summary[col] = display_summary[col].map(
                lambda x: f"{x:,.2f}" if np.isfinite(float(x)) else "∞"
            )
        st.dataframe(display_summary, use_container_width=True, hide_index=True)

        st.subheader("3. Critical Separation Location")
        selected_well = st.selectbox(
            "Select Offset Well",
            phase3_summary["Offset Well"].astype(str).tolist(),
            index=0,
            key="phase3_selected_well",
        )
        selected = phase3_summary[
            phase3_summary["Offset Well"].astype(str) == selected_well
        ].iloc[0]

        st.write(
            f"Critical separation occurs at **TVDSS {selected['TVDSS (m)']:.2f} m**: "
            f"**New Well MD {selected['New Well MD (m)']:.2f} m** vs "
            f"**{selected_well} MD {selected['Offset Well MD (m)']:.2f} m**, "
            f"with center-to-center distance **{selected['Center Distance (m)']:.2f} m**."
        )

        st.subheader("4. Interactive 3D — Preliminary Anti-Collision")

        # Focus-depth slider controls display only; it does not change the analysis.
        if not uresult.empty:
            z_all = pd.to_numeric(uresult["Z"], errors="coerce").dropna()
            if not z_all.empty:
                z_min = int(math.floor(float(z_all.min())))
                z_max = int(math.ceil(float(z_all.max())))
                if z_min == z_max:
                    z_max = z_min + 1
                focus_low, focus_high = st.slider(
                    "Focus Depth — TVDSS (m)",
                    min_value=z_min,
                    max_value=z_max,
                    value=(z_min, z_max),
                    step=5,
                    key="phase3_focus_tvdss",
                )
            else:
                focus_low, focus_high = -999999, 999999
        else:
            focus_low, focus_high = -999999, 999999

        st.caption(
            "Focus Depth only controls the visible TVDSS range. Anti-collision results "
            "are still calculated over the full screening interval."
        )

        fig_ac3 = go.Figure()
        if uresult.empty:
            st.error("Uncertainty result tidak tersedia untuk visualisasi.")
        else:
            new_ref = uresult[
                uresult["Well"].astype(str).str.strip().str.lower() == "new well"
            ].copy()
            offset_ref = uresult[
                uresult["Well"].astype(str).str.strip().str.lower() != "new well"
            ].copy()

            new_surface_e = float(new_ref.iloc[0]["X"])
            new_surface_n = float(new_ref.iloc[0]["Y"])
            new_surface_z = float(new_ref.iloc[0]["Z"])

            color_map = build_well_color_map(offset_ref)

            # All offset trajectories.
            for well, odf in offset_ref.groupby("Well"):
                odf = odf.sort_values("MD")
                odf = odf[(odf["Z"] >= focus_low) & (odf["Z"] <= focus_high)]
                if odf.empty:
                    continue
                fig_ac3.add_trace(
                    go.Scatter3d(
                        x=(odf["X"] - new_surface_e).astype(float),
                        y=(odf["Y"] - new_surface_n).astype(float),
                        z=(odf["Z"] - new_surface_z).astype(float),
                        mode="lines",
                        name=str(well),
                        line=dict(width=4, color=color_map.get(str(well))),
                        hovertemplate=(
                            f"<b>{well}</b><br>"
                            "MD=%{customdata[0]:.1f} m<br>"
                            "TVDSS=%{customdata[1]:.1f} m<br>"
                            "ΔEasting=%{x:.2f} m<br>"
                            "ΔNorthing=%{y:.2f} m<extra></extra>"
                        ),
                        customdata=odf[["MD", "Z"]].astype(float).to_numpy(),
                    )
                )

            # New Well trajectory.
            new_plot = new_ref.sort_values("MD")
            new_plot = new_plot[(new_plot["Z"] >= focus_low) & (new_plot["Z"] <= focus_high)]
            fig_ac3.add_trace(
                go.Scatter3d(
                    x=(new_plot["X"] - new_surface_e).astype(float),
                    y=(new_plot["Y"] - new_surface_n).astype(float),
                    z=(new_plot["Z"] - new_surface_z).astype(float),
                    mode="lines",
                    name="New Well",
                    line=dict(width=7, color="#ff4b4b"),
                    hovertemplate=(
                        "<b>New Well</b><br>"
                        "MD=%{customdata[0]:.1f} m<br>"
                        "TVDSS=%{customdata[1]:.1f} m<br>"
                        "ΔEasting=%{x:.2f} m<br>"
                        "ΔNorthing=%{y:.2f} m<extra></extra>"
                    ),
                    customdata=new_plot[["MD", "Z"]].astype(float).to_numpy(),
                )
            )

            # Continuous positional-error envelope tubes for every well.
            tube_sources = [("New Well", new_ref)] + [(str(w), g.copy()) for w, g in offset_ref.groupby("Well")]
            tube_colors = {"New Well": "#ff4b4b", **color_map}
            for tube_name, tube_df in tube_sources:
                path = interpolated_uncertainty_path(tube_df, step_md=5.0)
                path = path[(path["Z"] >= focus_low) & (path["Z"] <= focus_high)].copy()
                if len(path) < 2:
                    continue
                verts, ti, tj, tk, mesh_path = error_envelope_tube_from_path(path, n_ring=16)
                fig_ac3.add_trace(go.Mesh3d(
                    x=[v[0] - new_surface_e for v in verts],
                    y=[v[1] - new_surface_n for v in verts],
                    z=[v[2] - new_surface_z for v in verts],
                    i=ti, j=tj, k=tk,
                    name=f"{tube_name} Error Envelope",
                    legendgroup=f"envelope_{tube_name}",
                    showlegend=False,
                    color=tube_colors.get(tube_name, "#999999"),
                    opacity=0.16,
                    hoverinfo="skip",
                ))

            # Build critical rows for the selected pair.
            new_row = {
                "X": float(selected["New X"]),
                "Y": float(selected["New Y"]),
                "Z": float(selected["New Z"]),
                "_tx": float(selected["New TX"]),
                "_ty": float(selected["New TY"]),
                "_tz": float(selected["New TZ"]),
                "Azimuth Error": float(selected["New Well Azimuth Error (m)"]),
                "Inclination Error": float(selected["New Well Inclination Error (m)"]),
            }
            off_row = {
                "X": float(selected["Offset X"]),
                "Y": float(selected["Offset Y"]),
                "Z": float(selected["Offset Z"]),
                "_tx": float(selected["Offset TX"]),
                "_ty": float(selected["Offset TY"]),
                "_tz": float(selected["Offset TZ"]),
                "Azimuth Error": float(selected["Offset Well Azimuth Error (m)"]),
                "Inclination Error": float(selected["Offset Well Inclination Error (m)"]),
            }

            from trajectory import error_ellipse_from_row
            new_ellipse = error_ellipse_from_row(new_row, n_points=48)
            off_ellipse = error_ellipse_from_row(off_row, n_points=48)

            # Critical separation line.
            nx = float(selected["New X"] - new_surface_e)
            ny = float(selected["New Y"] - new_surface_n)
            nz = float(selected["New Z"] - new_surface_z)
            ox = float(selected["Offset X"] - new_surface_e)
            oy = float(selected["Offset Y"] - new_surface_n)
            oz = float(selected["Offset Z"] - new_surface_z)
            tvdss = float(selected["TVDSS (m)"])

            fig_ac3.add_trace(
                go.Scatter3d(
                    x=[nx, ox], y=[ny, oy], z=[nz, oz],
                    mode="lines",
                    name="Critical Separation",
                    line=dict(width=7, dash="dash", color="#ffffff"),
                    hovertemplate=(
                        f"<b>Critical Separation</b><br>"
                        f"TVDSS={tvdss:.2f} m<br>"
                        f"Distance={float(selected['Center Distance (m)']):.2f} m<extra></extra>"
                    ),
                )
            )

            # New Well critical error envelope.
            ex = [p[0] - new_surface_e for p in new_ellipse]
            ey = [p[1] - new_surface_n for p in new_ellipse]
            ez = [p[2] - new_surface_z for p in new_ellipse]
            fig_ac3.add_trace(
                go.Scatter3d(
                    x=ex, y=ey, z=ez,
                    mode="lines",
                    name="New Well Error Envelope",
                    showlegend=False,
                    line=dict(width=5, color="#ff4b4b"),
                    hovertemplate=(
                        f"<b>New Well Error Envelope</b><br>"
                        f"TVDSS={tvdss:.2f} m<br>"
                        f"Azimuth Error={float(selected['New Well Azimuth Error (m)']):.2f} m<br>"
                        f"Inclination Error={float(selected['New Well Inclination Error (m)']):.2f} m<br>"
                        f"Error Radius={float(selected['New Well Error Radius (m)']):.2f} m<extra></extra>"
                    ),
                )
            )

            # Offset critical error envelope.
            ex = [p[0] - new_surface_e for p in off_ellipse]
            ey = [p[1] - new_surface_n for p in off_ellipse]
            ez = [p[2] - new_surface_z for p in off_ellipse]
            fig_ac3.add_trace(
                go.Scatter3d(
                    x=ex, y=ey, z=ez,
                    mode="lines",
                    name=f"{selected_well} Error Envelope",
                    showlegend=False,
                    line=dict(width=5, color=color_map.get(selected_well, "#ffd166")),
                    hovertemplate=(
                        f"<b>{selected_well} Error Envelope</b><br>"
                        f"TVDSS={tvdss:.2f} m<br>"
                        f"Azimuth Error={float(selected['Offset Well Azimuth Error (m)']):.2f} m<br>"
                        f"Inclination Error={float(selected['Offset Well Inclination Error (m)']):.2f} m<br>"
                        f"Error Radius={float(selected['Offset Well Error Radius (m)']):.2f} m<extra></extra>"
                    ),
                )
            )

            # Critical points.
            fig_ac3.add_trace(
                go.Scatter3d(
                    x=[nx], y=[ny], z=[nz],
                    mode="markers",
                    name="New Well Critical Point",
                    marker=dict(size=9, color="#ff4b4b", symbol="diamond"),
                    hovertemplate=(
                        f"<b>New Well Critical Point</b><br>"
                        f"MD={float(selected['New Well MD (m)']):.2f} m<br>"
                        f"TVDSS={tvdss:.2f} m<br>"
                        f"Error Radius={float(selected['New Well Error Radius (m)']):.2f} m<extra></extra>"
                    ),
                )
            )
            fig_ac3.add_trace(
                go.Scatter3d(
                    x=[ox], y=[oy], z=[oz],
                    mode="markers",
                    name="Offset Critical Point",
                    marker=dict(size=9, color="#ffd166", symbol="diamond"),
                    hovertemplate=(
                        f"<b>{selected_well} Critical Point</b><br>"
                        f"MD={float(selected['Offset Well MD (m)']):.2f} m<br>"
                        f"TVDSS={tvdss:.2f} m<br>"
                        f"Error Radius={float(selected['Offset Well Error Radius (m)']):.2f} m<extra></extra>"
                    ),
                )
            )

            fig_ac3.update_layout(
                title=dict(
                    text="3D VIEW — PRELIMINARY ANTI-COLLISION",
                    x=0,
                    xanchor="left",
                    font=dict(size=15),
                ),
                height=720,
                margin=dict(l=0, r=0, t=85, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.03,
                    xanchor="left",
                    x=0,
                    font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    entrywidth=110,
                    itemsizing="constant",
                ),
                scene=dict(
                    xaxis_title="ΔEasting (m)",
                    yaxis_title="ΔNorthing (m)",
                    zaxis_title="TVDSS relative to New Well Surface (m)",
                    aspectmode="data",
                    camera=dict(eye=dict(x=1.55, y=1.55, z=1.25)),
                ),
            )
            st.plotly_chart(
                fig_ac3,
                use_container_width=True,
                config={"displaylogo": False},
            )

        # Engineering screening plots: one line per Offset Well, using New Well MD
        # as the common X-axis so the evolution of CtC and SF can be reviewed
        # directly against the planned New Well trajectory.
        if not phase3_details.empty:
            ctc_col, sf_col = st.columns(2)

            plot_df = phase3_details.copy()
            plot_df["New Well MD (m)"] = pd.to_numeric(plot_df["New Well MD (m)"], errors="coerce")
            plot_df["Center Distance (m)"] = pd.to_numeric(plot_df["Center Distance (m)"], errors="coerce")
            plot_df = plot_df.dropna(subset=["New Well MD (m)", "Center Distance (m)"])

            with ctc_col:
                st.subheader("5. CtC Plot")
                fig_ctc = go.Figure()
                for well, g in plot_df.groupby("Offset Well", sort=False):
                    g = g.sort_values("New Well MD (m)")
                    hover = np.column_stack([
                        g["Offset Well MD (m)"].astype(float).to_numpy(),
                        g["TVDSS (m)"].astype(float).to_numpy(),
                        g["Center Distance (m)"].astype(float).to_numpy(),
                    ])
                    fig_ctc.add_trace(go.Scatter(
                        x=g["New Well MD (m)"],
                        y=g["Center Distance (m)"],
                        mode="lines",
                        name=str(well),
                        customdata=hover,
                        hovertemplate=(
                            f"<b>{well}</b><br>"
                            "New Well MD=%{x:.2f} m<br>"
                            "Offset Well MD=%{customdata[0]:.2f} m<br>"
                            "TVDSS=%{customdata[1]:.2f} m<br>"
                            "CtC=%{y:.2f} m<extra></extra>"
                        ),
                        line=dict(width=2.5, color=color_map.get(str(well))),
                    ))
                # Highlight and label the single minimum CtC across all offset wells.
                min_idx = plot_df["Center Distance (m)"].idxmin()
                min_row = plot_df.loc[min_idx]
                min_well = str(min_row["Offset Well"])
                fig_ctc.add_trace(go.Scatter(
                    x=[min_row["New Well MD (m)"]],
                    y=[min_row["Center Distance (m)"]],
                    mode="markers+text",
                    text=[f'Min CtC = {min_row["Center Distance (m)"]:.2f} m'],
                    textposition="top center",
                    marker=dict(size=8, color=color_map.get(min_well)),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{min_well}</b><br>"
                        f"Min CtC={min_row['Center Distance (m)']:.2f} m<br>"
                        f"New Well MD={min_row['New Well MD (m)']:.2f} m<br>"
                        f"Offset Well MD={min_row['Offset Well MD (m)']:.2f} m<br>"
                        f"TVDSS={min_row['TVDSS (m)']:.2f} m<extra></extra>"
                    ),
                ))
                fig_ctc.update_layout(
                    title=dict(text="CENTER-TO-CENTER DISTANCE", x=0, xanchor="left"),
                    height=430,
                    margin=dict(l=55, r=15, t=55, b=55),
                    xaxis_title="New Well MD (m)",
                    yaxis_title="CtC (m)",
                    yaxis=dict(range=[0, 100]),
                    hovermode="closest",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                st.plotly_chart(fig_ctc, use_container_width=True, config={"displaylogo": False})

            with sf_col:
                st.subheader("6. SF Plot")
                fig_sf = go.Figure()
                sf_df = phase3_details.copy()
                sf_df["New Well MD (m)"] = pd.to_numeric(sf_df["New Well MD (m)"], errors="coerce")
                sf_df["Separation Factor"] = pd.to_numeric(sf_df["Separation Factor"], errors="coerce")
                sf_df = sf_df.dropna(subset=["New Well MD (m)", "Separation Factor"])
                for well, g in sf_df.groupby("Offset Well", sort=False):
                    g = g.sort_values("New Well MD (m)")
                    hover = np.column_stack([
                        g["Offset Well MD (m)"].astype(float).to_numpy(),
                        g["TVDSS (m)"].astype(float).to_numpy(),
                        g["Separation Factor"].astype(float).to_numpy(),
                    ])
                    fig_sf.add_trace(go.Scatter(
                        x=g["New Well MD (m)"],
                        y=g["Separation Factor"],
                        mode="lines",
                        name=str(well),
                        customdata=hover,
                        hovertemplate=(
                            f"<b>{well}</b><br>"
                            "New Well MD=%{x:.2f} m<br>"
                            "Offset Well MD=%{customdata[0]:.2f} m<br>"
                            "TVDSS=%{customdata[1]:.2f} m<br>"
                            "SF=%{y:.3f}<extra></extra>"
                        ),
                        line=dict(width=2.5, color=color_map.get(str(well))),
                    ))
                # Highlight and label the single minimum SF across all offset wells.
                min_idx = sf_df["Separation Factor"].idxmin()
                min_row = sf_df.loc[min_idx]
                min_well = str(min_row["Offset Well"])
                fig_sf.add_trace(go.Scatter(
                    x=[min_row["New Well MD (m)"]],
                    y=[min_row["Separation Factor"]],
                    mode="markers+text",
                    text=[f'Min SF = {min_row["Separation Factor"]:.2f}'],
                    textposition="top center",
                    marker=dict(size=8, color=color_map.get(min_well)),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{min_well}</b><br>"
                        f"Min SF={min_row['Separation Factor']:.3f}<br>"
                        f"New Well MD={min_row['New Well MD (m)']:.2f} m<br>"
                        f"Offset Well MD={min_row['Offset Well MD (m)']:.2f} m<br>"
                        f"TVDSS={min_row['TVDSS (m)']:.2f} m<extra></extra>"
                    ),
                ))

                # Screening reference levels. Each level uses a distinct visual cue.
                fig_sf.add_hline(y=1.0, line_dash="dash", line_color="#d62728", annotation_text="SF = 1.0", annotation_position="top left")
                fig_sf.add_hline(y=1.5, line_dash="dash", line_color="#ff7f0e", annotation_text="SF = 1.5", annotation_position="top left")
                fig_sf.add_hline(y=3.0, line_dash="dash", line_color="#2ca02c", annotation_text="SF = 3.0", annotation_position="top left")
                fig_sf.update_layout(
                    title=dict(text="SEPARATION FACTOR", x=0, xanchor="left"),
                    height=430,
                    margin=dict(l=55, r=15, t=55, b=55),
                    xaxis_title="New Well MD (m)",
                    yaxis_title="SF",
                    yaxis=dict(range=[0, 10]),
                    hovermode="closest",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                st.plotly_chart(fig_sf, use_container_width=True, config={"displaylogo": False})

        st.subheader("7. Detailed Screening Results")
        if not phase3_details.empty:
            detail_display = phase3_details.copy()
            detail_cols = [
                "TVDSS (m)",
                "New Well MD (m)",
                "Offset Well MD (m)",
                "Center Distance (m)",
                "New Well Azimuth Error (m)",
                "New Well Inclination Error (m)",
                "New Well Error Radius (m)",
                "Offset Well Azimuth Error (m)",
                "Offset Well Inclination Error (m)",
                "Offset Well Error Radius (m)",
                "Separation Factor",
            ]
            # Keep Offset Well first, followed by engineering values.
            shown_cols = ["Offset Well"] + detail_cols
            shown_cols = [c for c in shown_cols if c in detail_display.columns]
            detail_display = detail_display[shown_cols]
            for col in shown_cols[1:]:
                detail_display[col] = detail_display[col].map(
                    lambda x: f"{x:,.2f}" if np.isfinite(float(x)) else "∞"
                )
            st.dataframe(detail_display, use_container_width=True, hide_index=True)

            ac_csv = phase3_details.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Anti-Collision Screening CSV",
                data=ac_csv,
                file_name="preliminary_anti_collision_screening.csv",
                mime="text/csv",
                key="phase3_download_csv",
            )

            st.divider()
            st.subheader("8. Report")
            st.write("Enter the project/report information below, then generate the formal PDF report.")

            with st.expander("Project / Report Information", expanded=True):
                pi1, pi2 = st.columns(2)
                with pi1:
                    project_field = st.text_input("Project / Field Name", value="KKI Make-up Well Patuha 2026", key="project_field")
                    project_client = st.text_input("Client / Company", value="", key="project_client")
                    project_location = st.text_input("Location", value="Patuha, West Java", key="project_location")
                with pi2:
                    project_prepared = st.text_input("Prepared By", value="Rigsis Drilling Team", key="project_prepared")
                    project_new_well = st.text_input("New Well Name", value="PTH-V-7D", key="project_new_well")
                    project_report_no = st.text_input("Report Number", value="3", key="project_report_no")
                    project_revision = st.text_input("Revision", value="0", key="project_revision")

            include_survey_attachment = st.checkbox(
                "Include detailed survey data as an attachment in the PDF",
                value=True,
                key="include_survey_attachment",
            )

            if st.button("Generate Preliminary Anti-Collision PDF Report", type="primary", use_container_width=True, key="generate_pdf_report"):
                try:
                    # Pull the actual Phase 2 design inputs saved when the trajectory was calculated.
                    p3_saved_inputs = st.session_state.get("p3_inputs", {})
                    report_project = {
                        "field": project_field,
                        "client": project_client,
                        "location": project_location,
                        "prepared_by": project_prepared,
                        "report_number": project_report_no,
                        "revision": project_revision,
                        "new_well": project_new_well,
                        "offset_rate_azi": st.session_state.get("offset_rate_azi", 0.0),
                        "offset_rate_inc": st.session_state.get("offset_rate_inc", 0.0),
                        "new_rate_azi": st.session_state.get("new_rate_azi", 0.0),
                        "new_rate_inc": st.session_state.get("new_rate_inc", 0.0),
                        "ground_level": p3_saved_inputs.get("Ground Level (mASL)", "—"),
                        "rig_elevation": p3_saved_inputs.get("Rig Elevation (m)", "—"),
                        "rkb_elevation": p3_saved_inputs.get("RKB Elevation (mASL)", "—"),
                    }
                    # New Well trajectory values used by the report.
                    if not p3_detailed.empty:
                        report_project.update({
                            "new_well": project_new_well,
                            "target_n": p3_saved_inputs.get("Target Northing (m)", float(p3_detailed["Northing"].iloc[-1])),
                            "target_e": p3_saved_inputs.get("Target Easting (m)", float(p3_detailed["Easting"].iloc[-1])),
                            "target_depth": p3_saved_inputs.get("Target Depth (mASL)", float(p3_detailed["TVDSS"].iloc[-1])),
                        })
                    pdf_bytes = generate_pdf_report(
                        report_project,
                        st.session_state["result"],
                        p3_detailed,
                        uresult,
                        phase3_summary,
                        phase3_details,
                        include_survey_attachment=include_survey_attachment,
                    )
                    st.session_state["pdf_report_bytes"] = pdf_bytes
                    st.success("PDF report berhasil dibuat.")
                except Exception as exc:
                    st.error(f"PDF report tidak dapat dibuat: {exc}")

            if "pdf_report_bytes" in st.session_state:
                st.download_button(
                    "Download PDF Report",
                    data=st.session_state["pdf_report_bytes"],
                    file_name="preliminary_anti_collision_assessment.pdf",
                    mime="application/pdf",
                    key="download_pdf_report",
                    use_container_width=True,
                )

    st.caption(
        "This prototype is for preliminary screening only. Final anti-collision "
        "verification, survey error model and operational acceptance remain the "
        "responsibility of the directional drilling contractor."
    )

