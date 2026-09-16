"""
Offset Well Trajectory Engine
Phase 2 - Preliminary Anti-Collision Tool

Engineering method:
- Minimum Curvature Method
- Input: MD, Inclination, Azimuth
- Surface position: X, Y, Z
- Output: X, Y, Z, TVD, DLS
- Phase 2: depth-dependent uncertainty envelope
  using linear interpolation between surface and TD error.
"""

import math
import numpy as np
import pandas as pd


def _deg(value):
    return math.radians(float(value))


def minimum_curvature(survey_df, x0=0.0, y0=0.0, z0=0.0):
    """Calculate well trajectory using Minimum Curvature Method."""
    df = survey_df.copy()
    df = df.sort_values("MD").reset_index(drop=True)

    required = ["MD", "Inclination", "Azimuth"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing survey columns: {', '.join(missing)}")

    if df["MD"].duplicated().any():
        raise ValueError("Duplicate MD values found.")
    if (df["MD"].diff().dropna() <= 0).any():
        raise ValueError("MD must increase continuously.")
    if ((df["Inclination"] < 0) | (df["Inclination"] > 180)).any():
        raise ValueError("Inclination must be between 0 and 180 degrees.")
    if ((df["Azimuth"] < 0) | (df["Azimuth"] > 360)).any():
        raise ValueError("Azimuth must be between 0 and 360 degrees.")

    out = df.copy()
    out["X"] = float(x0)
    out["Y"] = float(y0)
    out["TVD"] = 0.0
    out["DLS"] = 0.0
    out["Z"] = float(z0)  # Z = surface elevation - TVD

    for i in range(1, len(out)):
        md1, md2 = out.loc[i-1, "MD"], out.loc[i, "MD"]
        inc1, inc2 = _deg(out.loc[i-1, "Inclination"]), _deg(out.loc[i, "Inclination"])
        azi1, azi2 = _deg(out.loc[i-1, "Azimuth"]), _deg(out.loc[i, "Azimuth"])
        dl = md2 - md1

        dogleg_angle = math.acos(max(-1.0, min(1.0,
            math.cos(inc2-inc1) -
            math.sin(inc1)*math.sin(inc2)*(1-math.cos(azi2-azi1))
        )))

        if abs(dogleg_angle) < 1e-12:
            rf = 1.0
        else:
            rf = 2.0 / dogleg_angle * math.tan(dogleg_angle / 2.0)

        d_tvd = (dl / 2.0) * (
            math.cos(inc1) + math.cos(inc2)
        ) * rf

        d_y = (dl / 2.0) * (
            math.sin(inc1)*math.cos(azi1) +
            math.sin(inc2)*math.cos(azi2)
        ) * rf

        d_x = (dl / 2.0) * (
            math.sin(inc1)*math.sin(azi1) +
            math.sin(inc2)*math.sin(azi2)
        ) * rf

        dls = math.degrees(dogleg_angle) / dl * 30.0 if dl else 0.0

        out.loc[i, "TVD"] = out.loc[i-1, "TVD"] + d_tvd
        out.loc[i, "X"] = out.loc[i-1, "X"] + d_x
        out.loc[i, "Y"] = out.loc[i-1, "Y"] + d_y
        out.loc[i, "DLS"] = dls
        out.loc[i, "Z"] = float(z0) - out.loc[i, "TVD"]

    return out


def calculate_all_wells(header_df, survey_df):
    """Calculate trajectories for all wells in the uploaded workbook."""
    required_h = ["Well", "X", "Y", "Z"]
    required_s = ["Well", "MD", "Inclination", "Azimuth"]

    for col in required_h:
        if col not in header_df.columns:
            raise ValueError(f"Well_Header is missing column: {col}")
    for col in required_s:
        if col not in survey_df.columns:
            raise ValueError(f"Survey is missing column: {col}")

    results = []
    for well in survey_df["Well"].dropna().astype(str).unique():
        h = header_df[header_df["Well"].astype(str) == well]
        if h.empty:
            raise ValueError(f"No X/Y/Z surface coordinates found for {well}.")
        row = h.iloc[0]
        s = survey_df[survey_df["Well"].astype(str) == well]
        calc = minimum_curvature(s, row["X"], row["Y"], row["Z"])

        # Well is already present in the survey data.
        calc["Well"] = well
        cols = ["Well"] + [c for c in calc.columns if c != "Well"]
        calc = calc[cols]
        results.append(calc)

    if not results:
        raise ValueError("No well survey data found.")

    return pd.concat(results, ignore_index=True)


def add_depth_dependent_uncertainty(
    result, surface_h=None, td_h=None, surface_v=None, td_v=None, error_params=None
):
    """Add positional uncertainty from an error rate per 1000 m MD.

    If ``error_params`` is supplied, it must contain: 
      - ``offset``: (azimuth_rate, inclination_rate) in m/1000 m MD
      - ``new``:    (azimuth_rate, inclination_rate) in m/1000 m MD

    The uncertainty grows linearly with MD from the start of each well.
    The legacy positional arguments are retained for compatibility, but when
    ``error_params`` is supplied the rate-based method is used.
    """
    out = result.copy()

    # New method: error rate per 1000 m MD.
    if error_params:
        offset_params = error_params.get("offset", (0.0, 0.0))
        new_params = error_params.get("new", offset_params)

        def _validate_rate(params):
            if len(params) != 2:
                raise ValueError("Error rate parameters must contain azimuth and inclination rates.")
            if any(float(v) < 0 for v in params):
                raise ValueError("Error rates cannot be negative.")
            return tuple(float(v) for v in params)

        offset_rate_azi, offset_rate_inc = _validate_rate(offset_params)
        new_rate_azi, new_rate_inc = _validate_rate(new_params)

        out["Azimuth Error"] = 0.0
        out["Inclination Error"] = 0.0

        for well, idx in out.groupby("Well").groups.items():
            rows = list(idx)
            md = out.loc[rows, "MD"].astype(float)
            md0 = float(md.min())
            md_from_start = (md - md0).clip(lower=0.0)

            well_key = str(well).strip().lower()
            rate_azi, rate_inc = (
                (new_rate_azi, new_rate_inc)
                if well_key == "new well"
                else (offset_rate_azi, offset_rate_inc)
            )

            out.loc[rows, "Azimuth Error"] = rate_azi * md_from_start / 1000.0
            out.loc[rows, "Inclination Error"] = rate_inc * md_from_start / 1000.0

        return out

    # Legacy surface-to-TD interpolation retained for backward compatibility.
    if any(v is None for v in (surface_h, td_h, surface_v, td_v)):
        raise ValueError("Uncertainty parameters are incomplete.")
    if any(float(v) < 0 for v in (surface_h, td_h, surface_v, td_v)):
        raise ValueError("Uncertainty values cannot be negative.")

    out["Azimuth Error"] = 0.0
    out["Inclination Error"] = 0.0
    for well, idx in out.groupby("Well").groups.items():
        rows = list(idx)
        md = out.loc[rows, "MD"].astype(float)
        md0 = float(md.min())
        mdtd = float(md.max())
        frac = pd.Series(0.0, index=md.index) if mdtd <= md0 else (md - md0) / (mdtd - md0)
        out.loc[rows, "Azimuth Error"] = float(surface_h) + frac * (float(td_h) - float(surface_h))
        out.loc[rows, "Inclination Error"] = float(surface_v) + frac * (float(td_v) - float(surface_v))
    return out


def _unit(v):
    n = math.sqrt(sum(x*x for x in v))
    if n < 1e-12:
        return (0.0, 0.0, 1.0)
    return tuple(x / n for x in v)


def _cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )


def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _local_horizontal_axes(tangent):
    """
    Create two horizontal directions tied to the trajectory azimuth.
    The first axis follows the horizontal projection of the trajectory.
    If the well is nearly vertical, global X is used as a stable reference.
    """
    tx, ty, _ = tangent
    hnorm = math.hypot(tx, ty)

    if hnorm < 1e-10:
        axis_a = (1.0, 0.0, 0.0)
    else:
        axis_a = (tx / hnorm, ty / hnorm, 0.0)

    axis_b = (-axis_a[1], axis_a[0], 0.0)
    return axis_a, axis_b



def _local_tangent(df, i):
    """Estimate the local wellbore tangent from neighbouring survey points."""
    n = len(df)

    if n <= 1:
        return (0.0, 0.0, -1.0)

    if i == 0:
        p0 = (float(df.loc[0, "X"]), float(df.loc[0, "Y"]), float(df.loc[0, "Z"]))
        p1 = (float(df.loc[1, "X"]), float(df.loc[1, "Y"]), float(df.loc[1, "Z"]))
        v = tuple(p1[k] - p0[k] for k in range(3))
    elif i == n - 1:
        p0 = (float(df.loc[n-2, "X"]), float(df.loc[n-2, "Y"]), float(df.loc[n-2, "Z"]))
        p1 = (float(df.loc[n-1, "X"]), float(df.loc[n-1, "Y"]), float(df.loc[n-1, "Z"]))
        v = tuple(p1[k] - p0[k] for k in range(3))
    else:
        pm = (float(df.loc[i-1, "X"]), float(df.loc[i-1, "Y"]), float(df.loc[i-1, "Z"]))
        pp = (float(df.loc[i+1, "X"]), float(df.loc[i+1, "Y"]), float(df.loc[i+1, "Z"]))
        v = tuple(pp[k] - pm[k] for k in range(3))

    return _unit(v)



def _local_ellipse_basis(df, i):
    """
    Build two orthogonal directions for a 2D positional-error ellipse.

    The ellipse is a 2D object. Its plane is normal to the local wellbore
    tangent, so the ellipse follows the hole direction in 3D.

    The major axis is Azimuth Error and the minor axis is Inclination Error.
    """
    tangent = _local_tangent(df, i)
    tx, ty, tz = tangent
    hnorm = math.hypot(tx, ty)

    if hnorm > 1e-10:
        # Local transverse horizontal direction.
        e1 = (-ty / hnorm, tx / hnorm, 0.0)
        e1 = _unit(e1)
    else:
        # For a near-vertical hole, use global X as reference.
        e1 = (1.0, 0.0, 0.0)

    # Second in-plane direction. Together e1/e2 form a plane perpendicular
    # to the wellbore tangent.
    e2 = _unit(_cross(tangent, e1))

    return e1, e2


def error_ellipse_at_station(df, station_index, n_points=40):
    """
    Create a closed 2D positional-error ellipse in 3D.

    Major semi-axis = Azimuth Error / 2
    Minor semi-axis = Inclination Error / 2

    The input values are treated as full ellipse width/length, consistent
    with the Phase 2 UI wording.
    """
    df = df.sort_values("MD").reset_index(drop=True)
    i = int(station_index)

    center = (
        float(df.loc[i, "X"]),
        float(df.loc[i, "Y"]),
        float(df.loc[i, "Z"]),
    )

    # User inputs represent full P x L dimensions.
    major = max(0.0, float(df.loc[i, "Azimuth Error"])) / 2.0
    minor = max(0.0, float(df.loc[i, "Inclination Error"])) / 2.0

    e1, e2 = _local_ellipse_basis(df, i)

    points = []
    for k in range(n_points + 1):
        theta = 2.0 * math.pi * k / n_points
        ct, st = math.cos(theta), math.sin(theta)

        x = center[0] + major * ct * e1[0] + minor * st * e2[0]
        y = center[1] + major * ct * e1[1] + minor * st * e2[1]
        z = center[2] + major * ct * e1[2] + minor * st * e2[2]

        points.append((x, y, z))

    return points


def select_ellipsoid_indices(df, start_md, end_md, interval_md):
    """
    Select survey stations for display only.

    Uncertainty is still calculated at every survey station. This function
    only controls which stations receive a visible ellipsoid.
    """
    df = df.sort_values("MD").reset_index(drop=True)

    if df.empty:
        return []

    start_md = float(start_md)
    end_md = float(end_md)
    interval_md = float(interval_md)

    if end_md < start_md:
        start_md, end_md = end_md, start_md

    # Keep stations inside the requested focus range.
    in_range = [
        i for i, md in enumerate(df["MD"].astype(float))
        if start_md <= md <= end_md
    ]
    if not in_range:
        return []

    # Choose the nearest actual survey station to each requested interval.
    target_mds = []
    target = start_md
    while target <= end_md + 1e-9:
        target_mds.append(target)
        target += interval_md

    selected = set()

    for target_md in target_mds:
        nearest = min(
            in_range,
            key=lambda idx: abs(float(df.loc[idx, "MD"]) - target_md)
        )
        selected.add(nearest)

    # Always include the first and last visible survey station in the focus.
    selected.add(in_range[0])
    selected.add(in_range[-1])

    return sorted(selected)



def _ellipse_axes_at_row(row):
    """Return ellipse semi-axes and local in-plane basis for one row."""
    a = max(0.0, float(row.get("Azimuth Error", 0.0))) / 2.0
    b = max(0.0, float(row.get("Inclination Error", 0.0))) / 2.0
    return a, b


def _interpolated_common_z(df, z_levels):
    """Interpolate one well onto common absolute-Z check planes.

    The common planes are shared by all wells. A well only contributes where
    the requested Z lies between its surface and TD elevations.
    """
    work = df.sort_values("MD").reset_index(drop=True).copy()
    if work.empty:
        return pd.DataFrame()

    # Z should normally decrease with MD. Use ascending Z for np.interp.
    base_z = work["Z"].astype(float).to_numpy()
    order = np.argsort(base_z)
    base_z = base_z[order]
    # Remove duplicate Z values so interpolation is well-defined.
    keep = np.r_[True, np.diff(base_z) > 1e-10]
    base_z = base_z[keep]
    source = work.iloc[order].reset_index(drop=True).loc[keep].reset_index(drop=True)
    if len(base_z) == 0:
        return pd.DataFrame()

    zmin = float(base_z.min())
    zmax = float(base_z.max())
    levels = [float(z) for z in z_levels if zmin - 1e-9 <= float(z) <= zmax + 1e-9]
    if not levels:
        return pd.DataFrame()

    out = pd.DataFrame({"Z": levels})
    numeric_cols = [c for c in [
        "MD", "X", "Y", "TVD", "Azimuth Error", "Inclination Error"
    ] if c in source.columns]
    for col in numeric_cols:
        out[col] = np.interp(
            out["Z"].to_numpy(),
            base_z,
            source[col].astype(float).to_numpy(),
        )

    # Local wellbore tangent at each common-Z plane.
    xyz = out[["X", "Y", "Z"]].astype(float).to_numpy()
    tangents = []
    for i in range(len(out)):
        if len(out) == 1:
            v = np.array([0.0, 0.0, -1.0])
        elif i == 0:
            v = xyz[1] - xyz[0]
        elif i == len(out) - 1:
            v = xyz[-1] - xyz[-2]
        else:
            v = xyz[i + 1] - xyz[i - 1]
        n = np.linalg.norm(v)
        tangents.append(v / n if n > 1e-12 else np.array([0.0, 0.0, -1.0]))

    out["_tx"] = [v[0] for v in tangents]
    out["_ty"] = [v[1] for v in tangents]
    out["_tz"] = [v[2] for v in tangents]
    return out


def _ellipse_axes_at_row(row):
    """Return ellipse semi-axes for one common-Z station."""
    a = max(0.0, float(row.get("Azimuth Error", 0.0))) / 2.0
    b = max(0.0, float(row.get("Inclination Error", 0.0))) / 2.0
    return a, b


def _ellipse_radius_in_direction(row, direction):
    """Directional semi-axis radius of the station ellipse."""
    a, b = _ellipse_axes_at_row(row)
    tangent = np.array([
        float(row["_tx"]), float(row["_ty"]), float(row["_tz"])
    ])
    dn = np.linalg.norm(tangent)
    if dn < 1e-12:
        tangent = np.array([0.0, 0.0, -1.0])
    else:
        tangent = tangent / dn

    hnorm = math.hypot(tangent[0], tangent[1])
    if hnorm > 1e-10:
        e1 = np.array([-tangent[1] / hnorm, tangent[0] / hnorm, 0.0])
    else:
        e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(tangent, e1)
    e2n = np.linalg.norm(e2)
    if e2n > 1e-12:
        e2 = e2 / e2n

    u = np.asarray(direction, dtype=float)
    un = np.linalg.norm(u)
    if un < 1e-12:
        return max(a, b)
    u = u / un
    return float(math.sqrt((a * np.dot(u, e1))**2 + (b * np.dot(u, e2))**2))


def error_ellipse_from_row(row, n_points=40):
    """Create a 2D positional-error ellipse from an interpolated TVDSS row."""
    center = (float(row["X"]), float(row["Y"]), float(row["Z"]))
    major = max(0.0, float(row.get("Azimuth Error", 0.0))) / 2.0
    minor = max(0.0, float(row.get("Inclination Error", 0.0))) / 2.0
    tangent = np.array([float(row["_tx"]), float(row["_ty"]), float(row["_tz"])], dtype=float)
    tn = np.linalg.norm(tangent)
    tangent = tangent / tn if tn > 1e-12 else np.array([0.0, 0.0, -1.0])
    hnorm = math.hypot(tangent[0], tangent[1])
    if hnorm > 1e-10:
        e1 = np.array([-tangent[1] / hnorm, tangent[0] / hnorm, 0.0])
    else:
        e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(tangent, e1)
    e2n = np.linalg.norm(e2)
    if e2n > 1e-12:
        e2 = e2 / e2n
    pts = []
    for k in range(n_points + 1):
        theta = 2.0 * math.pi * k / n_points
        ct, st = math.cos(theta), math.sin(theta)
        v = np.array(center) + major * ct * e1 + minor * st * e2
        pts.append((float(v[0]), float(v[1]), float(v[2])))
    return pts


def preliminary_anti_collision(new_df, offset_df, interval_z=5.0, start_offset=25.0):
    """Preliminary 3D screening on common absolute-Z planes.

    The first check plane is 25 m below the highest well surface elevation.
    Subsequent planes are every 5 m downward and are shared by every well.
    Each well is interpolated to the same absolute Z before the center-to-
    center distance and directional uncertainty are compared.
    """
    all_wells = pd.concat([new_df, offset_df], ignore_index=True)
    if all_wells.empty:
        return pd.DataFrame(), pd.DataFrame()

    highest_surface_z = float(all_wells.groupby("Well")["Z"].first().max())
    deepest_z = float(all_wells["Z"].min())
    first_z = highest_surface_z - float(start_offset)
    step = float(interval_z)
    if step <= 0:
        raise ValueError("Interval Z harus lebih besar dari 0 m.")
    if first_z < deepest_z - 1e-9:
        return pd.DataFrame(), pd.DataFrame()

    z_levels = list(np.arange(first_z, deepest_z - 1e-9, -step))
    if not z_levels:
        z_levels = [first_z]
    if z_levels[-1] > deepest_z + 1e-9:
        z_levels.append(deepest_z)

    new = _interpolated_common_z(new_df, z_levels)
    offsets = {
        str(well): _interpolated_common_z(df, z_levels)
        for well, df in offset_df.groupby("Well")
    }

    rows = []
    details = []
    for well, off in offsets.items():
        if new.empty or off.empty:
            continue

        # Only compare at Z planes where both wells physically exist.
        common = pd.merge(
            new, off, on="Z", how="inner", suffixes=("_new", "_off")
        )
        if common.empty:
            continue

        min_record = None
        for _, r in common.iterrows():
            p_new = np.array([r["X_new"], r["Y_new"], r["Z"]], dtype=float)
            p_off = np.array([r["X_off"], r["Y_off"], r["Z"]], dtype=float)
            dvec = p_off - p_new
            dist = float(np.linalg.norm(dvec))
            direction = dvec if dist > 1e-12 else np.array([1.0, 0.0, 0.0])

            new_row = {"_tx": r["_tx_new"], "_ty": r["_ty_new"], "_tz": r["_tz_new"],
                       "Azimuth Error": r.get("Azimuth Error_new", 0.0),
                       "Inclination Error": r.get("Inclination Error_new", 0.0)}
            off_row = {"_tx": r["_tx_off"], "_ty": r["_ty_off"], "_tz": r["_tz_off"],
                       "Azimuth Error": r.get("Azimuth Error_off", 0.0),
                       "Inclination Error": r.get("Inclination Error_off", 0.0)}
            r_new = _ellipse_radius_in_direction(new_row, direction)
            r_off = _ellipse_radius_in_direction(off_row, -direction)
            combined = r_new + r_off
            sf = dist / combined if combined > 1e-12 else float("inf")

            record = {
                "Offset Well": well,
                "TVDSS (m)": float(r["Z"]),
                "New Well MD (m)": float(r["MD_new"]),
                "Offset Well MD (m)": float(r["MD_off"]),
                "Center Distance (m)": dist,
                "New Well Azimuth Error (m)": float(r.get("Azimuth Error_new", 0.0)),
                "New Well Inclination Error (m)": float(r.get("Inclination Error_new", 0.0)),
                "New Well Error Radius (m)": r_new,
                "Offset Well Azimuth Error (m)": float(r.get("Azimuth Error_off", 0.0)),
                "Offset Well Inclination Error (m)": float(r.get("Inclination Error_off", 0.0)),
                "Offset Well Error Radius (m)": r_off,
                "Combined Error Radius (m)": combined,
                "Separation Factor": sf,
                "New X": float(r["X_new"]), "New Y": float(r["Y_new"]), "New Z": float(r["Z"]),
                "Offset X": float(r["X_off"]), "Offset Y": float(r["Y_off"]), "Offset Z": float(r["Z"]),
                "New TX": float(r["_tx_new"]), "New TY": float(r["_ty_new"]), "New TZ": float(r["_tz_new"]),
                "Offset TX": float(r["_tx_off"]), "Offset TY": float(r["_ty_off"]), "Offset TZ": float(r["_tz_off"]),
            }
            details.append({k: record[k] for k in [
                "Offset Well", "TVDSS (m)", "New Well MD (m)", "Offset Well MD (m)",
                "Center Distance (m)",
                "New Well Azimuth Error (m)", "New Well Inclination Error (m)",
                "New Well Error Radius (m)",
                "Offset Well Azimuth Error (m)", "Offset Well Inclination Error (m)",
                "Offset Well Error Radius (m)",
                "Separation Factor"
            ]})
            if min_record is None or sf < min_record["Separation Factor"]:
                min_record = record

        if min_record is not None:
            sf = min_record["Separation Factor"]
            if sf < 1.0:
                status = "ENVELOPE OVERLAP - MODIFY TRAJECTORY"
            elif sf < 1.5:
                status = "CRITICAL - CONSIDER TO MODIFY TRAJECTORY"
            elif sf < 3.0:
                status = "CLOSE - REVIEW TRAJECTORY"
            else:
                status = "PRELIMINARY PASS"
            min_record["Screening Status"] = status
            rows.append(min_record)

    summary = pd.DataFrame(rows)
    details_df = pd.DataFrame(details)
    if summary.empty:
        return summary, details_df
    summary = summary.sort_values("Separation Factor").reset_index(drop=True)
    return summary, details_df


def interpolated_uncertainty_path(df, step_md=5.0):
    """Create a regularly spaced trajectory path for continuous 3D error-envelope display.

    The path is interpolated every ``step_md`` along MD. Position and positional
    error are interpolated from the survey stations, while the local tangent is
    estimated from the interpolated XYZ path.
    """
    work = df.sort_values("MD").reset_index(drop=True).copy()
    if work.empty:
        return pd.DataFrame()
    md = work["MD"].astype(float).to_numpy()
    if len(md) == 1:
        targets = md.copy()
    else:
        step_md = max(float(step_md), 0.1)
        targets = np.arange(md[0], md[-1] + 1e-9, step_md)
        if targets[-1] < md[-1] - 1e-9:
            targets = np.append(targets, md[-1])

    out = pd.DataFrame({"MD": targets})
    for col in ["X", "Y", "Z", "TVD", "Azimuth Error", "Inclination Error"]:
        if col in work.columns:
            out[col] = np.interp(targets, md, work[col].astype(float).to_numpy())

    xyz = out[["X", "Y", "Z"]].astype(float).to_numpy()
    tangents = []
    for i in range(len(out)):
        if len(out) == 1:
            v = np.array([0.0, 0.0, -1.0])
        elif i == 0:
            v = xyz[1] - xyz[0]
        elif i == len(out) - 1:
            v = xyz[-1] - xyz[-2]
        else:
            v = xyz[i + 1] - xyz[i - 1]
        n = np.linalg.norm(v)
        tangents.append(v / n if n > 1e-12 else np.array([0.0, 0.0, -1.0]))

    out["_tx"] = [v[0] for v in tangents]
    out["_ty"] = [v[1] for v in tangents]
    out["_tz"] = [v[2] for v in tangents]
    return out


def error_envelope_tube(df, step_md=5.0, n_ring=16):
    """Generate a continuous uncertainty-envelope tube along a well trajectory.

    Each interpolated 5 m station becomes an elliptical cross-section. Adjacent
    cross-sections are connected into a triangular surface for interactive 3D use.
    """
    path = interpolated_uncertainty_path(df, step_md=step_md)
    if path.empty:
        return [], [], [], path

    n_ring = max(8, int(n_ring))
    vertices = []
    rings = []
    for _, row in path.iterrows():
        center = np.array([float(row["X"]), float(row["Y"]), float(row["Z"])])
        a = max(0.0, float(row.get("Azimuth Error", 0.0))) / 2.0
        b = max(0.0, float(row.get("Inclination Error", 0.0))) / 2.0
        tangent = np.array([float(row["_tx"]), float(row["_ty"]), float(row["_tz"])])
        tn = np.linalg.norm(tangent)
        tangent = tangent / tn if tn > 1e-12 else np.array([0.0, 0.0, -1.0])
        hnorm = math.hypot(tangent[0], tangent[1])
        if hnorm > 1e-10:
            e1 = np.array([-tangent[1] / hnorm, tangent[0] / hnorm, 0.0])
        else:
            e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.cross(tangent, e1)
        e2n = np.linalg.norm(e2)
        if e2n > 1e-12:
            e2 = e2 / e2n
        ring = []
        for k in range(n_ring):
            theta = 2.0 * math.pi * k / n_ring
            p = center + a * math.cos(theta) * e1 + b * math.sin(theta) * e2
            ring.append(len(vertices))
            vertices.append((float(p[0]), float(p[1]), float(p[2])))
        rings.append(ring)

    i_idx, j_idx, k_idx = [], [], []
    for r in range(len(rings) - 1):
        r0, r1 = rings[r], rings[r + 1]
        for k in range(n_ring):
            kn = (k + 1) % n_ring
            i_idx.extend([r0[k], r1[k]])
            j_idx.extend([r1[kn], r0[kn]])
            k_idx.extend([r0[kn], r1[kn]])

    return i_idx, j_idx, k_idx, path


def error_envelope_tube_from_path(path, n_ring=16):
    """Create a triangular tube mesh from an already interpolated uncertainty path.

    Returns vertices, triangle indices, and the path used for the mesh.
    """
    if path is None or path.empty:
        return [], [], [], [], path
    n_ring = max(8, int(n_ring))
    vertices = []
    rings = []
    for _, row in path.iterrows():
        center = np.array([float(row["X"]), float(row["Y"]), float(row["Z"])])
        a = max(0.0, float(row.get("Azimuth Error", 0.0))) / 2.0
        b = max(0.0, float(row.get("Inclination Error", 0.0))) / 2.0
        tangent = np.array([float(row["_tx"]), float(row["_ty"]), float(row["_tz"])])
        tn = np.linalg.norm(tangent)
        tangent = tangent / tn if tn > 1e-12 else np.array([0.0, 0.0, -1.0])
        hnorm = math.hypot(tangent[0], tangent[1])
        if hnorm > 1e-10:
            e1 = np.array([-tangent[1] / hnorm, tangent[0] / hnorm, 0.0])
        else:
            e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.cross(tangent, e1)
        e2n = np.linalg.norm(e2)
        if e2n > 1e-12:
            e2 = e2 / e2n
        ring = []
        for k in range(n_ring):
            theta = 2.0 * math.pi * k / n_ring
            p = center + a * math.cos(theta) * e1 + b * math.sin(theta) * e2
            ring.append(len(vertices))
            vertices.append((float(p[0]), float(p[1]), float(p[2])))
        rings.append(ring)

    i_idx, j_idx, k_idx = [], [], []
    for r in range(len(rings) - 1):
        r0, r1 = rings[r], rings[r + 1]
        for k in range(n_ring):
            kn = (k + 1) % n_ring
            i_idx.extend([r0[k], r1[k]])
            j_idx.extend([r1[kn], r0[kn]])
            k_idx.extend([r0[kn], r1[kn]])
    return vertices, i_idx, j_idx, k_idx, path
