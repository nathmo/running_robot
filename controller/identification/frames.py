"""Inertia-tensor comparison + frame alignment — PURE numpy (no mujoco/scipy).

This is the maths behind the Limbs & Inertia panel. A CAD inertia tensor can be wrong two ways:
its magnitudes are off, OR it is expressed in a differently-oriented (possibly arbitrary) reference
frame. The two are separable:

  * The EIGENVALUES of an inertia tensor (its principal moments) are frame-invariant, so comparing
    the sorted eigenvalues tells you whether the magnitudes are plausible REGARDLESS of orientation.
  * The EIGENVECTORS give the rotation between the two frames, so a best-fit rotation reveals a pure
    frame mismatch and lets the viewer realign one tensor onto the other.

So: eigenvalues match but orientation differs  -> "right values, wrong frame (rotate by X deg)";
eigenvalues far apart even after realignment    -> "values wrong / wrong file".
"""
import numpy as np

# eigenvalue ratio inside this band (both directions) => "magnitude plausible"
PLAUSIBLE_RATIO = 1.5
# best-fit rotation below this angle => the frames are effectively already aligned
ALIGNED_ANGLE_DEG = 12.0


def to_matrix(inertia):
    """{ixx,iyy,izz,ixy,ixz,iyz} (or a length-6 seq ixx,iyy,izz,ixy,ixz,iyz) -> symmetric 3x3."""
    if isinstance(inertia, dict):
        ixx, iyy, izz = inertia["ixx"], inertia["iyy"], inertia["izz"]
        ixy, ixz, iyz = inertia.get("ixy", 0.0), inertia.get("ixz", 0.0), inertia.get("iyz", 0.0)
    else:
        ixx, iyy, izz, ixy, ixz, iyz = inertia
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], float)


def to_dict(M):
    M = np.asarray(M, float)
    return {"ixx": float(M[0, 0]), "iyy": float(M[1, 1]), "izz": float(M[2, 2]),
            "ixy": float(M[0, 1]), "ixz": float(M[0, 2]), "iyz": float(M[1, 2])}


def principal(M):
    """Sorted-ascending eigenvalues + a right-handed eigenvector matrix (columns = principal axes)."""
    M = 0.5 * (np.asarray(M, float) + np.asarray(M, float).T)   # symmetrize defensively
    w, V = np.linalg.eigh(M)                                    # ascending, orthonormal
    if np.linalg.det(V) < 0:
        V[:, 0] = -V[:, 0]                                      # make it a proper rotation
    return w, V


def ellipsoid_semi_axes(M, mass):
    """Semi-axes (a,b,c) of the UNIFORM-DENSITY solid ellipsoid with the same mass + principal
    moments (a nice, honest way to *draw* an inertia tensor). For a solid ellipsoid,
    I1 = m/5 (b^2+c^2) etc., so a^2 = 5(I2+I3-I1)/(2m). Physical tensors satisfy the triangle
    inequality I_i <= I_j+I_k; clamp tiny negatives from numerical noise."""
    w, V = principal(M)
    if mass is None or mass <= 0:
        return np.array([0.0, 0.0, 0.0]), V
    s = w.sum()
    a2 = np.clip((s - 2.0 * w) * 5.0 / (2.0 * mass), 0.0, None)   # [a^2, b^2, c^2] per axis order
    return np.sqrt(a2), V


def best_fit_rotation(M_from, M_to):
    """Rotation R (det=+1) minimizing ||R M_from R^T - M_to||_F. Brute-forces the 3x3 eigenvector
    permutations + sign flips (48 candidates — trivial and degeneracy-proof), keeping the proper
    rotation with the smallest residual. Returns (R, angle_deg, residual_frac)."""
    A, B = to_matrix(M_from) if isinstance(M_from, dict) else np.asarray(M_from, float), \
           to_matrix(M_to) if isinstance(M_to, dict) else np.asarray(M_to, float)
    _, Ua = principal(A)
    _, Ub = principal(B)
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    signs = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
             (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]
    scale = max(np.linalg.norm(B), 1e-12)
    cands = []
    for p in perms:
        Ub_p = Ub[:, p]
        for s in signs:
            R = (Ub_p * np.array(s)) @ Ua.T
            if np.linalg.det(R) < 0:
                continue                                       # only proper rotations
            resid = np.linalg.norm(R @ A @ R.T - B) / scale
            angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
            cands.append((resid, angle, R))
    best_resid = min(c[0] for c in cands)
    # among rotations that fit equally well (degenerate eigenvalues / tensor symmetries admit many),
    # prefer the SMALLEST angle so an already-aligned tensor reports ~0deg, not an equivalent 180deg
    R, angle, resid = min(((R, a, r) for r, a, R in cands if r <= best_resid + 0.02),
                          key=lambda t: t[1])
    return R, float(angle), float(resid)


def compare(cad, ident):
    """Compare a CAD body {mass, com, inertia} with an identified one and classify the discrepancy.

    Returns a JSON-able dict with: sorted principal moments for each, their per-axis ratio, the
    best-fit CAD->identified rotation (matrix + angle), mass/CoM deltas, a plain-language verdict,
    and a suggested domain-randomization fraction (how much the model should be randomized given the
    CAD-vs-identified spread). Either side may be None (not yet identified)."""
    out = {"cad": _body_view(cad), "identified": _body_view(ident)}
    if cad is None or ident is None:
        out["verdict"] = "identified value not available yet" if ident is None else "no CAD value"
        out["category"] = "incomplete"
        return out

    Icad, Iid = to_matrix(cad["inertia"]), to_matrix(ident["inertia"])
    wc, _ = principal(Icad)
    wi, _ = principal(Iid)
    ratios = [float(b / a) if a > 1e-12 else None for a, b in zip(wc, wi)]
    R, angle, resid = best_fit_rotation(Icad, Iid)

    finite = [r for r in ratios if r is not None]
    worst = max(max(finite), 1.0 / min(finite)) if finite else float("inf")
    plausible = worst <= PLAUSIBLE_RATIO
    aligned = angle <= ALIGNED_ANGLE_DEG

    if plausible and aligned:
        category, verdict = "match", (
            "plausible — magnitudes agree and the frames are already aligned "
            f"(worst principal-moment ratio {worst:.2f}x, rotation {angle:.0f} deg)")
    elif plausible and not aligned:
        category, verdict = "wrong_frame", (
            f"right magnitudes, WRONG reference frame — rotate the CAD tensor by {angle:.0f} deg to "
            f"align (worst principal-moment ratio only {worst:.2f}x after realignment)")
    else:
        category, verdict = "wrong_values", (
            f"off even after realignment — principal moments differ by up to {worst:.1f}x "
            "(likely wrong values / wrong CAD file for this body)")

    mass_c = (cad.get("mass") or 0.0)
    mass_i = (ident.get("mass") or 0.0)
    out.update(
        principal_moments={"cad": [float(v) for v in wc], "identified": [float(v) for v in wi],
                           "ratio": ratios},
        rotation={"angle_deg": angle, "matrix": R.tolist(), "residual_frac": resid},
        mass={"cad": mass_c, "identified": mass_i,
              "ratio": float(mass_i / mass_c) if mass_c > 1e-9 else None},
        com_offset_m=_com_offset(cad.get("com"), ident.get("com")),
        verdict=verdict, category=category,
        suggested_dr_frac=_suggested_dr(ratios, mass_c, mass_i))
    return out


def _body_view(b):
    if b is None:
        return None
    M = to_matrix(b["inertia"])
    w, V = principal(M)
    axes, _ = ellipsoid_semi_axes(M, b.get("mass"))
    return {"mass": b.get("mass"), "com": list(b["com"]) if b.get("com") is not None else None,
            "inertia": to_dict(M), "principal_moments": [float(v) for v in w],
            "principal_axes": V.tolist(), "ellipsoid_semi_axes": [float(v) for v in axes]}


def _com_offset(a, b):
    if a is None or b is None:
        return None
    d = np.asarray(b, float) - np.asarray(a, float)
    return {"vector": d.tolist(), "norm": float(np.linalg.norm(d))}


def _suggested_dr(ratios, mass_c, mass_i):
    """Heuristic domain-randomization fraction: the largest relative gap between CAD and identified
    (principal moments + mass). If CAD already matches, a small floor keeps some robustness margin."""
    gaps = [abs(r - 1.0) for r in ratios if r is not None]
    if mass_c > 1e-9 and mass_i > 0:
        gaps.append(abs(mass_i / mass_c - 1.0))
    return round(float(min(max(max(gaps, default=0.1), 0.05), 0.6)), 3)
