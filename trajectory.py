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
        offset_params = error_params.get("offset", (25.0, 15.0))
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


# ----------------------------------------------------------------------
# New-well design upload and nudging
# ----------------------------------------------------------------------

PHASE2_SUMMARY_COLUMNS = [
    "Parameter", "MD", "TVD", "Inc", "Azimuth",
    "N+", "E+", "Northing", "Easting",
    "Displacement", "TVDSS", "BUR",
]

_NUDGE_TOOLFACE = {
    "build": 0.0,
    "right": 90.0,
    "drop": 180.0,
    "left": 270.0,
}


def _wrap180(angle_deg):
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _dogleg_deg(inc1, azi1, inc2, azi2):
    i1 = math.radians(float(inc1))
    i2 = math.radians(float(inc2))
    da = math.radians(_wrap180(float(azi2) - float(azi1)))
    value = math.cos(i1) * math.cos(i2) + math.sin(i1) * math.sin(i2) * math.cos(da)
    value = max(-1.0, min(1.0, value))
    return math.degrees(math.acos(value))


def _interp_inc_azi(mds, incs, azis, md):
    md = float(md)
    if md <= float(mds[0]):
        return float(incs[0]), float(azis[0]) % 360.0
    if md >= float(mds[-1]):
        return float(incs[-1]), float(azis[-1]) % 360.0
    i = int(np.searchsorted(mds, md) - 1)
    span = float(mds[i + 1] - mds[i])
    t = 0.0 if span <= 1e-12 else (md - float(mds[i])) / span
    inc = float(incs[i]) + t * (float(incs[i + 1]) - float(incs[i]))
    azi = (float(azis[i]) + t * _wrap180(float(azis[i + 1]) - float(azis[i]))) % 360.0
    return inc, azi


def phase2_frame_from_calculated(calc_df, surface_easting, surface_northing):
    """Map a minimum-curvature result into the Phase 2 detailed-survey columns.

    Minimum curvature stores X as Easting and Y as Northing.
    """
    calc = calc_df.reset_index(drop=True)
    n_plus = calc["Y"].astype(float) - float(surface_northing)
    e_plus = calc["X"].astype(float) - float(surface_easting)
    bur = calc["DLS"].astype(float) if "DLS" in calc.columns else 0.0
    return pd.DataFrame({
        "MD": calc["MD"].astype(float),
        "TVD": calc["TVD"].astype(float),
        "Inclination": calc["Inclination"].astype(float),
        "Azimuth": calc["Azimuth"].astype(float),
        "N+": n_plus,
        "E+": e_plus,
        "Northing": calc["Y"].astype(float),
        "Easting": calc["X"].astype(float),
        "Displacement": np.hypot(n_plus.to_numpy(), e_plus.to_numpy()),
        "TVDSS": calc["Z"].astype(float),
        "BUR": bur,
        "Parameter": "Survey",
    })


def phase2_summary_from_frame(detailed, markers):
    """Build the Phase 2 summary table from labeled measured depths."""
    rows = []
    for label, md in markers:
        idx = (detailed["MD"] - float(md)).abs().idxmin()
        r = detailed.loc[idx]
        rows.append([
            label,
            float(r["MD"]), float(r["TVD"]), float(r["Inclination"]), float(r["Azimuth"]),
            float(r["N+"]), float(r["E+"]), float(r["Northing"]), float(r["Easting"]),
            float(r["Displacement"]), float(r["TVDSS"]), float(r["BUR"]),
        ])
    return pd.DataFrame(rows, columns=PHASE2_SUMMARY_COLUMNS)


def load_design_well(header_df, survey_df, well_name):
    """Load one well from the offset-well workbook and calculate its path.

    Well_Header columns are Well, X, Y, Z. X is Easting, Y is Northing and
    Z is the surface elevation used as the TVD datum. Survey columns are
    Well, MD, Inclination, Azimuth. The path uses minimum curvature.
    """
    header = header_df.copy()
    survey = survey_df.copy()
    header.columns = [str(c).strip() for c in header.columns]
    survey.columns = [str(c).strip() for c in survey.columns]

    missing_h = [c for c in ("Well", "X", "Y", "Z") if c not in header.columns]
    missing_s = [c for c in ("Well", "MD", "Inclination", "Azimuth") if c not in survey.columns]
    if missing_h or missing_s:
        raise ValueError(
            "Workbook does not match the offset-well template. "
            f"Well_Header missing: {', '.join(missing_h) or 'none'}. "
            f"Survey missing: {', '.join(missing_s) or 'none'}."
        )

    well_name = str(well_name).strip()
    header["Well"] = header["Well"].astype(str).str.strip()
    survey["Well"] = survey["Well"].astype(str).str.strip()
    header_row = header[header["Well"] == well_name]
    survey = survey[survey["Well"] == well_name].copy()
    if header_row.empty:
        raise ValueError(f"No Well_Header row for {well_name}.")
    if survey.empty:
        raise ValueError(f"No Survey rows for {well_name}.")

    try:
        surface_e = float(header_row.iloc[0]["X"])
        surface_n = float(header_row.iloc[0]["Y"])
        surface_z = float(header_row.iloc[0]["Z"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Well_Header X, Y and Z must be numbers.") from exc

    for col in ("MD", "Inclination", "Azimuth"):
        survey[col] = pd.to_numeric(survey[col], errors="coerce")
    survey = survey.dropna(subset=["MD", "Inclination", "Azimuth"])
    if survey.empty:
        raise ValueError(f"Survey rows for {well_name} have no numeric MD, inclination and azimuth.")

    calc = minimum_curvature(survey, surface_e, surface_n, surface_z)
    detailed = phase2_frame_from_calculated(calc, surface_e, surface_n)
    detailed.loc[detailed.index[0], "Parameter"] = "RKB"
    detailed.loc[detailed.index[-1], "Parameter"] = "TD"
    base_survey = calc[["MD", "Inclination", "Azimuth"]].copy()
    return detailed, base_survey, (surface_e, surface_n, surface_z)


def _steer_step(inc, azi, delta_md, bur, toolface_deg):
    """Advance one station at a constant dogleg severity and toolface."""
    kappa = math.radians(float(bur)) / 30.0
    tf = math.radians(float(toolface_deg))
    new_inc = float(inc) + math.degrees(kappa * delta_md * math.cos(tf))
    turning = abs(math.sin(tf)) > 1e-8
    if turning and min(float(inc), new_inc) <= 0.5:
        raise ValueError(
            "A left or right nudge needs inclination above about 0.5°. "
            "Use Build to create inclination before turning."
        )
    if new_inc <= 0.05 or new_inc >= 179.95:
        raise ValueError(
            "Nudge would drive inclination outside 0–180°. "
            "Reduce the BUR or the nudge length, or start where inclination is already established."
        )
    sin_avg = math.sin(math.radians((float(inc) + new_inc) / 2.0))
    d_azi = 0.0 if sin_avg < 1e-8 else math.degrees(kappa * delta_md * math.sin(tf) / sin_avg)
    new_azi = (float(azi) + d_azi) % 360.0
    return new_inc, new_azi


def nudge_survey(
    survey_df,
    start_md,
    nudge_length,
    return_length,
    bur,
    direction,
    step_md=5.0,
):
    """Steer off the design survey, then restore the original attitude.

    ``direction`` is one of ``build``, ``drop``, ``left`` or ``right``.
    The nudge holds the requested BUR for ``nudge_length`` metres starting
    at ``start_md``. The return section steers back so inclination and
    azimuth match the original survey at the end of the return. Stations
    below that depth keep the original inclination and azimuth, which
    leaves the deeper well parallel to the design.

    Returns a survey with columns MD, Inclination and Azimuth.
    """
    direction_key = str(direction).strip().lower()
    if direction_key not in _NUDGE_TOOLFACE:
        raise ValueError("Nudge direction must be build, drop, left or right.")

    work = survey_df.copy()
    required = ["MD", "Inclination", "Azimuth"]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"Survey is missing columns: {', '.join(missing)}")
    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=required).sort_values("MD").drop_duplicates("MD").reset_index(drop=True)
    if len(work) < 2:
        raise ValueError("At least two survey stations are required to nudge a trajectory.")

    start_md = float(start_md)
    nudge_length = float(nudge_length)
    return_length = float(return_length)
    bur = float(bur)
    step_md = float(step_md)
    if nudge_length <= 0 or return_length <= 0:
        raise ValueError("Nudge length and return length must be greater than 0 m.")
    if bur <= 0:
        raise ValueError("BUR must be greater than 0 deg/30 m.")
    if step_md <= 0:
        raise ValueError("Nudge step must be greater than 0 m.")

    mds = work["MD"].to_numpy(dtype=float)
    incs = work["Inclination"].to_numpy(dtype=float)
    azis = work["Azimuth"].to_numpy(dtype=float)
    td = float(mds[-1])
    nudge_end = start_md + nudge_length
    return_end = nudge_end + return_length
    if start_md < float(mds[0]) - 1e-6:
        raise ValueError(f"Nudge start MD must be at or below the first station ({mds[0]:.2f} m).")
    if start_md >= td - 1e-6:
        raise ValueError("Nudge start MD must be above TD.")
    if return_end > td + 1e-6:
        raise ValueError(
            f"Nudge plus return ends at MD {return_end:.2f} m, past TD {td:.2f} m. "
            "Shorten the nudge or extend TD."
        )

    def _md_steps(md0, md1):
        count = max(1, int(math.ceil((md1 - md0) / step_md - 1e-9)))
        return [md0 + (md1 - md0) * (i + 1) / count for i in range(count)]

    stations = []
    for md, inc, azi in zip(mds, incs, azis):
        if float(md) < start_md - 1e-6:
            stations.append((float(md), float(inc), float(azi) % 360.0))
    inc, azi = _interp_inc_azi(mds, incs, azis, start_md)
    stations.append((start_md, inc, azi))

    toolface = _NUDGE_TOOLFACE[direction_key]
    for md in _md_steps(start_md, nudge_end):
        prev_md, inc, azi = stations[-1]
        inc, azi = _steer_step(inc, azi, md - prev_md, bur, toolface)
        stations.append((md, inc, azi))

    target_inc, target_azi = _interp_inc_azi(mds, incs, azis, return_end)
    away_inc, away_azi = stations[-1][1], stations[-1][2]
    required = _dogleg_deg(away_inc, away_azi, target_inc, target_azi)
    available = bur * return_length / 30.0
    if required > available + 0.05:
        raise ValueError(
            f"Return cannot restore the original attitude. Required dogleg is {required:.2f}°, "
            f"but {return_length:.1f} m at {bur:.2f}°/30 m can deliver {available:.2f}°. "
            "Increase the return length or the BUR."
        )

    for md in _md_steps(nudge_end, return_end):
        prev_md, inc, azi = stations[-1]
        remain = return_end - prev_md
        step = md - prev_md
        if remain <= step + 1e-8:
            stations.append((return_end, target_inc, target_azi % 360.0))
            break
        hs = math.radians(target_inc - inc)
        sin_avg = math.sin(math.radians(max((inc + target_inc) / 2.0, 0.2)))
        rs = sin_avg * math.radians(_wrap180(target_azi - azi))
        phi = math.hypot(hs, rs)
        if phi < 1e-10:
            stations.append((md, target_inc, target_azi % 360.0))
            continue
        kappa = min(phi / remain, math.radians(bur) / 30.0)
        tf = math.atan2(rs, hs)
        new_inc = inc + math.degrees(kappa * step * math.cos(tf))
        if new_inc <= 0.05 or new_inc >= 179.95:
            raise ValueError(
                "Return would drive inclination outside 0–180°. "
                "Increase the return length or reduce the nudge."
            )
        sin_step = math.sin(math.radians((inc + new_inc) / 2.0))
        d_azi = 0.0 if sin_step < 1e-8 else math.degrees(kappa * step * math.sin(tf) / sin_step)
        stations.append((md, new_inc, (azi + d_azi) % 360.0))

    for md, inc, azi in zip(mds, incs, azis):
        if float(md) > return_end + 1e-6:
            stations.append((float(md), float(inc), float(azi) % 360.0))

    out = pd.DataFrame(stations, columns=["MD", "Inclination", "Azimuth"])
    out["MD"] = out["MD"].round(6)
    out = out.drop_duplicates("MD", keep="last").sort_values("MD").reset_index(drop=True)
    if (out["MD"].diff().dropna() <= 0).any():
        raise ValueError("Nudge produced a survey whose MD does not increase.")
    return out


# ----------------------------------------------------------------------
# Nudge, then connect to the main J-type design
# ----------------------------------------------------------------------

_NUDGE_CLOSURE_M = 0.5


def _dogleg_rad_arrays(inc1, azi1, inc2, azi2):
    """Dogleg in radians. Angles are in radians. Inputs may be arrays."""
    val = (
        np.cos(inc1) * np.cos(inc2)
        + np.sin(inc1) * np.sin(inc2) * np.cos(azi2 - azi1)
    )
    return np.arccos(np.clip(val, -1.0, 1.0))


def _arc_displacement(inc1, azi1, inc2, azi2, length):
    """North, East and TVD change along one minimum-curvature arc.

    Angles are radians. ``length`` is measured depth along the arc.
    """
    phi = _dogleg_rad_arrays(inc1, azi1, inc2, azi2)
    tiny = np.abs(phi) < 1e-10
    rf = np.ones_like(phi, dtype=float)
    rf = np.where(tiny, 1.0, (2.0 / np.where(tiny, 1.0, phi)) * np.tan(np.where(tiny, 0.0, phi) / 2.0))
    # A zero-length or zero-dogleg step follows the starting attitude.
    straight_n = length * np.sin(inc1) * np.cos(azi1)
    straight_e = length * np.sin(inc1) * np.sin(azi1)
    straight_v = length * np.cos(inc1)
    d_n = (length / 2.0) * (np.sin(inc1) * np.cos(azi1) + np.sin(inc2) * np.cos(azi2)) * rf
    d_e = (length / 2.0) * (np.sin(inc1) * np.sin(azi1) + np.sin(inc2) * np.sin(azi2)) * rf
    d_v = (length / 2.0) * (np.cos(inc1) + np.cos(inc2)) * rf
    use_straight = tiny | (np.abs(length) < 1e-12)
    return (
        np.where(use_straight, straight_n, d_n),
        np.where(use_straight, straight_e, d_e),
        np.where(use_straight, straight_v, d_v),
    )


def _attitude_along_arc(inc1, azi1, inc2, azi2, fraction):
    """Inclination and azimuth, in degrees, at a fraction of a constant-curvature arc."""
    i1, a1 = math.radians(inc1), math.radians(azi1)
    i2, a2 = math.radians(inc2), math.radians(azi2)
    t1 = np.array([math.sin(i1) * math.cos(a1), math.sin(i1) * math.sin(a1), math.cos(i1)])
    t2 = np.array([math.sin(i2) * math.cos(a2), math.sin(i2) * math.sin(a2), math.cos(i2)])
    phi = float(_dogleg_rad_arrays(i1, a1, i2, a2))
    s = min(1.0, max(0.0, float(fraction)))
    if phi < 1e-10:
        tangent = t1
    else:
        tangent = (math.sin((1.0 - s) * phi) * t1 + math.sin(s * phi) * t2) / math.sin(phi)
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-12:
        tangent = t1
    else:
        tangent = tangent / norm
    inc = math.degrees(math.acos(min(1.0, max(-1.0, float(tangent[2])))))
    azi = math.degrees(math.atan2(float(tangent[1]), float(tangent[0]))) % 360.0
    return inc, azi


def _nudge_section(start_md, nudge_inc_deg, nudge_azi_deg, nudge_bur):
    """Vertical hole to ``start_md``, then a build to the nudge inclination."""
    if start_md < -1e-6:
        raise ValueError("Nudge start depth must be 0 m or deeper.")
    if not 0.0 < float(nudge_inc_deg) < 90.0:
        raise ValueError("Nudge inclination must be between 0 and 90 degrees.")
    if float(nudge_bur) <= 0:
        raise ValueError("Nudge BUR must be greater than 0 deg/30 m.")
    radius = 30.0 / math.radians(float(nudge_bur))
    inc = math.radians(float(nudge_inc_deg))
    build_md = radius * inc
    end_md = float(start_md) + build_md
    tvd_end = float(start_md) + radius * math.sin(inc)
    hd_end = radius * (1.0 - math.cos(inc))
    azi = math.radians(float(nudge_azi_deg) % 360.0)
    return {
        "start_md": float(start_md),
        "end_md": end_md,
        "inc_deg": float(nudge_inc_deg),
        "azi_deg": float(nudge_azi_deg) % 360.0,
        "bur": float(nudge_bur),
        "radius": radius,
        "tvd_end": tvd_end,
        "hd_end": hd_end,
        "north_end": hd_end * math.cos(azi),
        "east_end": hd_end * math.sin(azi),
    }


def _hold_to_kop(nudge, kop):
    """Hold the nudge attitude from the end of the nudge down to KOP."""
    if float(kop) < nudge["end_md"] - 0.01:
        raise ValueError(
            f"KOP ({float(kop):.2f} m) is above the end of the nudge "
            f"({nudge['end_md']:.2f} m MD). The main-design KOP must be below the nudge."
        )
    kop = max(float(kop), nudge["end_md"])
    dmd = kop - nudge["end_md"]
    inc = math.radians(nudge["inc_deg"])
    azi = math.radians(nudge["azi_deg"])
    tvd = nudge["tvd_end"] + dmd * math.cos(inc)
    hd = nudge["hd_end"] + dmd * math.sin(inc)
    return {
        "kop": kop,
        "tvd": tvd,
        "north": hd * math.cos(azi),
        "east": hd * math.sin(azi),
        "inc": nudge["inc_deg"],
        "azi": nudge["azi_deg"],
    }


def _closure_residual(inc0, azi0, inc_f, azi_f, length, tangent_length, delta):
    d_n, d_e, d_v = _arc_displacement(
        math.radians(inc0), math.radians(azi0),
        math.radians(inc_f), math.radians(azi_f),
        float(length),
    )
    i_f = math.radians(inc_f)
    a_f = math.radians(azi_f)
    pred = np.array([
        float(d_n) + tangent_length * math.sin(i_f) * math.cos(a_f),
        float(d_e) + tangent_length * math.sin(i_f) * math.sin(a_f),
        float(d_v) + tangent_length * math.cos(i_f),
    ], dtype=float)
    miss = pred - np.asarray(delta, dtype=float)
    return float(np.linalg.norm(miss)), pred


def _best_curve_tangent(inc0, azi0, delta, inc_f, azi_f, bur):
    """Closure of one minimum-curvature curve at a fixed BUR, plus its tangent."""
    phi = float(_dogleg_rad_arrays(
        math.radians(inc0), math.radians(azi0),
        math.radians(inc_f), math.radians(azi_f),
    ))
    dls = math.radians(float(bur)) / 30.0
    curve_len = 0.0 if phi < 1e-10 else phi / dls
    d_n, d_e, d_v = _arc_displacement(
        math.radians(inc0), math.radians(azi0),
        math.radians(inc_f), math.radians(azi_f),
        curve_len,
    )
    remain = np.asarray(delta, dtype=float) - np.array([float(d_n), float(d_e), float(d_v)])
    tangent = np.array([
        math.sin(math.radians(inc_f)) * math.cos(math.radians(azi_f)),
        math.sin(math.radians(inc_f)) * math.sin(math.radians(azi_f)),
        math.cos(math.radians(inc_f)),
    ])
    tangent_len = float(np.dot(remain, tangent))
    if tangent_len < -1e-3:
        return None
    miss = float(np.linalg.norm(remain - tangent_len * tangent))
    return miss, float(inc_f), float(azi_f) % 360.0, curve_len, tangent_len


def _refine_fixed_bur(inc0, azi0, delta, inc_f, azi_f, bur, inc_fixed=False):
    """Tighten inclination and azimuth around a seed. Inclination stays fixed when requested."""
    best = _best_curve_tangent(inc0, azi0, delta, inc_f, azi_f, bur)
    if best is None:
        best = (1e9, inc_f, azi_f, 0.0, 0.0)
    for scale in (1.0, 0.2, 0.05):
        inc_vals = [inc_f] if inc_fixed else np.linspace(best[1] - scale, best[1] + scale, 9)
        azi_vals = np.linspace(best[2] - 2.0 * scale, best[2] + 2.0 * scale, 17)
        for inc_try in inc_vals:
            if not 0.2 < float(inc_try) < 89.8:
                continue
            for azi_try in azi_vals:
                item = _best_curve_tangent(inc0, azi0, delta, float(inc_try), float(azi_try) % 360.0, bur)
                if item is not None and item[0] < best[0]:
                    best = item
    return best


def _solve_curve_and_tangent(inc0, azi0, delta, final_inc=None, bur=None):
    """Find the build/turn and tangent that land on ``delta`` (north, east, TVD).

    Supply ``final_inc`` to solve BUR, or ``bur`` to solve the final inclination.
    Returns final inc, final azimuth, curve length, tangent length, BUR and closure.
    """
    delta = np.asarray(delta, dtype=float)
    if final_inc is not None and bur is None:
        inc_f = float(final_inc)
        i0 = math.radians(inc0)
        a0 = math.radians(azi0)
        i_f = math.radians(inc_f)
        best = None
        for azi in np.linspace(0.0, 360.0, 721, endpoint=False):
            a_f = math.radians(float(azi))
            phi = float(_dogleg_rad_arrays(i0, a0, i_f, a_f))
            rf = 1.0 if phi < 1e-10 else (2.0 / phi) * math.tan(phi / 2.0)
            matrix = np.array([
                [0.5 * (math.sin(i0) * math.cos(a0) + math.sin(i_f) * math.cos(a_f)) * rf,
                 math.sin(i_f) * math.cos(a_f)],
                [0.5 * (math.sin(i0) * math.sin(a0) + math.sin(i_f) * math.sin(a_f)) * rf,
                 math.sin(i_f) * math.sin(a_f)],
                [0.5 * (math.cos(i0) + math.cos(i_f)) * rf, math.cos(i_f)],
            ], dtype=float)
            sol, _, _, _ = np.linalg.lstsq(matrix, delta, rcond=None)
            curve_len = float(sol[0])
            tangent_len = float(sol[1])
            if curve_len < -1e-3 or tangent_len < -1e-3:
                continue
            curve_len = max(0.0, curve_len)
            tangent_len = max(0.0, tangent_len)
            miss, _ = _closure_residual(inc0, azi0, inc_f, float(azi), curve_len, tangent_len, delta)
            phi_deg = math.degrees(phi)
            solved_bur = 0.0 if curve_len < 1e-4 else phi_deg / curve_len * 30.0
            item = (miss, inc_f, float(azi) % 360.0, curve_len, tangent_len, solved_bur)
            if best is None or item[0] < best[0]:
                best = item
        if best is None:
            raise ValueError(
                "The target cannot be reached from the nudge at this inclination. "
                "Lower the nudge inclination or move the KOP."
            )
        seed_azi = best[2]
        for azi in np.linspace(seed_azi - 0.6, seed_azi + 0.6, 121):
            a_f = math.radians(float(azi) % 360.0)
            phi = float(_dogleg_rad_arrays(i0, a0, i_f, a_f))
            rf = 1.0 if phi < 1e-10 else (2.0 / phi) * math.tan(phi / 2.0)
            matrix = np.array([
                [0.5 * (math.sin(i0) * math.cos(a0) + math.sin(i_f) * math.cos(a_f)) * rf,
                 math.sin(i_f) * math.cos(a_f)],
                [0.5 * (math.sin(i0) * math.sin(a0) + math.sin(i_f) * math.sin(a_f)) * rf,
                 math.sin(i_f) * math.sin(a_f)],
                [0.5 * (math.cos(i0) + math.cos(i_f)) * rf, math.cos(i_f)],
            ], dtype=float)
            sol, _, _, _ = np.linalg.lstsq(matrix, delta, rcond=None)
            curve_len = max(0.0, float(sol[0]))
            tangent_len = max(0.0, float(sol[1]))
            if float(sol[0]) < -1e-3 or float(sol[1]) < -1e-3:
                continue
            miss, _ = _closure_residual(inc0, azi0, inc_f, float(azi) % 360.0, curve_len, tangent_len, delta)
            phi_deg = math.degrees(phi)
            solved_bur = 0.0 if curve_len < 1e-4 else phi_deg / curve_len * 30.0
            item = (miss, inc_f, float(azi) % 360.0, curve_len, tangent_len, solved_bur)
            if item[0] < best[0]:
                best = item
        miss, inc_f, azi_f, curve_len, tangent_len, solved_bur = best
    elif bur is not None and final_inc is None:
        incs = np.linspace(0.5, 89.0, 178)
        azis = np.linspace(0.0, 360.0, 360, endpoint=False)
        best = None
        for inc_f in incs:
            for azi in azis[::2]:
                item = _best_curve_tangent(inc0, azi0, delta, float(inc_f), float(azi), bur)
                if item is not None and (best is None or item[0] < best[0]):
                    best = item
        if best is None:
            raise ValueError(
                f"BUR {float(bur):.2f} deg/30 m cannot build a path from the nudge to the target. "
                "Increase the BUR."
            )
        best = _refine_fixed_bur(inc0, azi0, delta, best[1], best[2], bur, inc_fixed=False)
        miss, inc_f, azi_f, curve_len, tangent_len = best
        solved_bur = float(bur)
    else:
        raise ValueError("The main design must fix either the final inclination or the BUR.")

    if miss > _NUDGE_CLOSURE_M:
        if bur is not None:
            raise ValueError(
                f"BUR {float(bur):.2f} deg/30 m is too low to reach the target from the nudge. "
                "Increase the BUR, or move the nudge so the main design has room to build and turn. "
                f"The closest approach is {miss:.2f} m."
            )
        raise ValueError(
            "The target cannot be reached with these values. "
            "The KOP has to be below the end of the nudge, and the build has to be able to turn "
            f"from the nudge ({inc0:.2f}° inclination, {azi0:.2f}° azimuth) onto the target. "
            f"The closest approach is {miss:.2f} m."
        )
    return {
        "inc": float(inc_f),
        "azimuth": float(azi_f) % 360.0,
        "curve_length": float(curve_len),
        "tangent_length": float(tangent_len),
        "bur": float(solved_bur),
        "closure_m": float(miss),
    }


def _solve_kop_inc_bur(nudge, target_north, target_east, target_tvd, target_inc, bur):
    """Solve KOP for a fixed final inclination and BUR. KOP stays below the nudge."""
    inc = math.radians(nudge["inc_deg"])
    if abs(math.cos(inc)) < 1e-4:
        raise ValueError("The nudge inclination is too close to horizontal to hold down to a KOP.")
    max_hold = max(0.0, (target_tvd - nudge["tvd_end"]) / math.cos(inc))
    kop_hi = nudge["end_md"] + max_hold
    if kop_hi < nudge["end_md"] + 1.0:
        raise ValueError(
            f"The nudge already reaches TVD {nudge['tvd_end']:.2f} m, at or below the target TVD "
            f"{target_tvd:.2f} m. Reduce the nudge start depth, inclination or BUR."
        )
    best = None
    for kop in np.linspace(nudge["end_md"], kop_hi, 120):
        held = _hold_to_kop(nudge, float(kop))
        if held["tvd"] > target_tvd + 0.5:
            continue
        delta = np.array([
            target_north - held["north"],
            target_east - held["east"],
            target_tvd - held["tvd"],
        ])
        for azi in np.linspace(0.0, 360.0, 360, endpoint=False):
            item = _best_curve_tangent(
                nudge["inc_deg"], nudge["azi_deg"], delta, float(target_inc), float(azi), bur
            )
            if item is None:
                continue
            packed = (item[0], float(kop), item[2], item[3], item[4])
            if best is None or packed[0] < best[0]:
                best = packed
    if best is None:
        raise ValueError(
            f"BUR {float(bur):.2f} deg/30 m is too low, or the nudge ends too deep, "
            "to reach the target with a KOP below the nudge."
        )
    miss, kop, azi, curve_len, tangent_len = best
    for kop_try in np.linspace(kop - 25.0, kop + 25.0, 26):
        if kop_try < nudge["end_md"] - 1e-6:
            continue
        held = _hold_to_kop(nudge, float(kop_try))
        delta = np.array([
            target_north - held["north"],
            target_east - held["east"],
            target_tvd - held["tvd"],
        ])
        refined = _refine_fixed_bur(
            nudge["inc_deg"], nudge["azi_deg"], delta, float(target_inc), azi, bur, inc_fixed=True
        )
        if refined[0] < miss:
            miss, _, azi, curve_len, tangent_len = refined
            kop = float(kop_try)
    if miss > _NUDGE_CLOSURE_M:
        raise ValueError(
            f"BUR {float(bur):.2f} deg/30 m is too low, or the nudge ends too deep, "
            "to reach the target with a KOP below the nudge. "
            f"The closest approach is {miss:.2f} m."
        )
    return {
        "kop": float(kop),
        "inc": float(target_inc),
        "azimuth": float(azi) % 360.0,
        "curve_length": float(curve_len),
        "tangent_length": float(tangent_len),
        "bur": float(bur),
        "closure_m": float(miss),
    }


def plan_nudged_design(
    surface_n, surface_e, rkb,
    target_n, target_e, target_tvdss,
    nudge_start_md, nudge_azimuth, nudge_inc, nudge_bur,
    method, kop=None, target_inc=None, bur=None,
    td_mode="Automatic", custom_td=None, station_interval=30.0,
):
    """Build a vertical hole, a nudge, then the main design on to the target.

    ``method`` is ``KOP + Inclination``, ``KOP + BUR`` or ``Inclination + BUR``.
    The main curve uses minimum curvature from the nudge attitude to the
    attitude that reaches the target. Returns the Phase 2 detailed frame,
    summary, info and the solved design values.
    """
    target_tvd = float(rkb) - float(target_tvdss)
    if target_tvd <= 0:
        raise ValueError("Target TVD must be below the RKB.")
    nudge = _nudge_section(nudge_start_md, nudge_inc, nudge_azimuth, nudge_bur)
    if nudge["tvd_end"] > target_tvd + 0.5:
        raise ValueError(
            f"The nudge reaches TVD {nudge['tvd_end']:.2f} m before the target TVD "
            f"{target_tvd:.2f} m. Reduce the nudge start depth, inclination or BUR."
        )

    rel_n = float(target_n) - float(surface_n)
    rel_e = float(target_e) - float(surface_e)

    if method == "KOP + Inclination":
        if kop is None or target_inc is None:
            raise ValueError("KOP and target inclination are required for this design method.")
        held = _hold_to_kop(nudge, kop)
        if held["tvd"] > target_tvd + 0.5:
            raise ValueError(
                f"KOP {held['kop']:.2f} m is too deep: holding the nudge reaches TVD "
                f"{held['tvd']:.2f} m before the target TVD {target_tvd:.2f} m."
            )
        solved = _solve_curve_and_tangent(
            nudge["inc_deg"], nudge["azi_deg"],
            np.array([rel_n - held["north"], rel_e - held["east"], target_tvd - held["tvd"]]),
            final_inc=float(target_inc),
        )
        solved["kop"] = held["kop"]
    elif method == "KOP + BUR":
        if kop is None or bur is None:
            raise ValueError("KOP and BUR are required for this design method.")
        if float(bur) <= 0:
            raise ValueError("BUR must be greater than 0 deg/30 m.")
        held = _hold_to_kop(nudge, kop)
        if held["tvd"] > target_tvd + 0.5:
            raise ValueError(
                f"KOP {held['kop']:.2f} m is too deep: holding the nudge reaches TVD "
                f"{held['tvd']:.2f} m before the target TVD {target_tvd:.2f} m."
            )
        solved = _solve_curve_and_tangent(
            nudge["inc_deg"], nudge["azi_deg"],
            np.array([rel_n - held["north"], rel_e - held["east"], target_tvd - held["tvd"]]),
            bur=float(bur),
        )
        solved["kop"] = held["kop"]
    elif method == "Inclination + BUR":
        if target_inc is None or bur is None:
            raise ValueError("Target inclination and BUR are required for this design method.")
        if float(bur) <= 0:
            raise ValueError("BUR must be greater than 0 deg/30 m.")
        solved = _solve_kop_inc_bur(
            nudge, rel_n, rel_e, target_tvd, float(target_inc), float(bur)
        )
    else:
        raise ValueError(f"Unknown design method: {method}")

    kop = float(solved["kop"])
    curve_len = float(solved["curve_length"])
    tangent_len = float(solved["tangent_length"])
    eob = kop + curve_len
    target_md = eob + tangent_len
    if td_mode == "Automatic":
        td = math.ceil(target_md / station_interval - 1e-12) * station_interval
    else:
        if custom_td is None:
            raise ValueError("Custom TD is required.")
        td = float(custom_td)
        if td < target_md - 1e-6:
            raise ValueError(f"Custom TD must be at least the target MD ({target_md:.2f} m).")

    keys = [0.0, nudge["start_md"], nudge["end_md"], kop, eob, target_md, td]
    regular = list(np.arange(0.0, td + station_interval * 0.01, station_interval))
    mds = sorted({
        round(float(md), 6)
        for md in regular + keys
        if -1e-8 <= float(md) <= td + 1e-6
    })
    rows = []
    for md in mds:
        if md <= nudge["start_md"] + 1e-8:
            inc, azi, section = 0.0, 0.0, "Vertical"
        elif md <= nudge["end_md"] + 1e-8:
            theta = (md - nudge["start_md"]) / nudge["radius"]
            inc, azi, section = math.degrees(theta), nudge["azi_deg"], "Nudge"
        elif md <= kop + 1e-8:
            inc, azi, section = nudge["inc_deg"], nudge["azi_deg"], "Hold"
        elif md <= eob + 1e-8 and curve_len > 1e-6:
            inc, azi = _attitude_along_arc(
                nudge["inc_deg"], nudge["azi_deg"],
                solved["inc"], solved["azimuth"],
                (md - kop) / curve_len,
            )
            section = "Build"
        else:
            inc, azi, section = solved["inc"], solved["azimuth"], "Tangent"
        rows.append({"MD": md, "Inclination": inc, "Azimuth": azi, "Section": section})
    survey = pd.DataFrame(rows)
    calc = minimum_curvature(survey[["MD", "Inclination", "Azimuth"]], surface_e, surface_n, rkb)
    detailed = phase2_frame_from_calculated(calc, surface_e, surface_n)
    detailed["Parameter"] = survey["Section"].to_numpy()
    detailed.loc[detailed.index[0], "Parameter"] = "RKB"
    target_idx = (detailed["MD"] - target_md).abs().idxmin()
    detailed.loc[target_idx, "Parameter"] = "Target"
    detailed.loc[target_idx, "Northing"] = float(target_n)
    detailed.loc[target_idx, "Easting"] = float(target_e)
    detailed.loc[target_idx, "TVDSS"] = float(target_tvdss)
    detailed.loc[target_idx, "TVD"] = target_tvd
    detailed.loc[target_idx, "N+"] = rel_n
    detailed.loc[target_idx, "E+"] = rel_e
    detailed.loc[target_idx, "Displacement"] = math.hypot(rel_n, rel_e)
    detailed.loc[target_idx, "Inclination"] = solved["inc"]
    detailed.loc[target_idx, "Azimuth"] = solved["azimuth"]
    detailed.loc[detailed.index[-1], "Parameter"] = "TD"

    markers = [
        ("RKB", 0.0),
        ("Nudge start", nudge["start_md"]),
        ("Nudge end", nudge["end_md"]),
        ("KOP", kop),
        ("EOB", eob),
        ("Target", target_md),
        ("TD", td),
    ]
    summary = phase2_summary_from_frame(detailed, markers)
    info = {
        "source": "Nudged design",
        "nudge_start_md": nudge["start_md"],
        "nudge_end_md": nudge["end_md"],
        "nudge_inc": nudge["inc_deg"],
        "nudge_azimuth": nudge["azi_deg"],
        "nudge_bur": nudge["bur"],
        "kop": kop,
        "eob": eob,
        "target_md": target_md,
        "td": td,
        "target_inc": solved["inc"],
        "bur": solved["bur"],
        "final_azimuth": solved["azimuth"],
        "closure_m": solved["closure_m"],
        "target_tvd": target_tvd,
        "rkb_elevation": float(rkb),
        "target_depth_masl": float(target_tvdss),
    }
    return {
        "survey": survey[["MD", "Inclination", "Azimuth"]].copy(),
        "detailed": detailed,
        "summary": summary,
        "info": info,
        "solved": solved,
        "surface": (float(surface_e), float(surface_n), float(rkb)),
    }
