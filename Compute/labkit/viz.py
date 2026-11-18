"""3-D viewers and animations.

py3Dmol (pip install py3Dmol) draws interactive WebGL molecules in Jupyter:
drag to rotate, scroll to zoom. If it is missing, a matplotlib animation is
used instead, so the notebooks still run.
"""
import numpy as np

CPK = {"H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
       "S": "#FFFF30", "F": "#90E050", "Cl": "#1FF01F", "D": "#FFFFC0"}
RADIUS = {"H": 0.31, "D": 0.31, "C": 0.70, "N": 0.65, "O": 0.60, "S": 1.00, "F": 0.50, "Cl": 0.99}


def _xyz_block(symbols, X, comment=""):
    lines = [str(len(symbols)), comment]
    for s, (x, y, z) in zip(symbols, np.asarray(X)):
        lines.append(f"{'H' if s == 'D' else s} {x:.5f} {y:.5f} {z:.5f}")
    return "\n".join(lines) + "\n"


def _have_py3dmol():
    try:
        import py3Dmol  # noqa: F401
        return True
    except ImportError:
        return False


def _style(view):
    view.setStyle({"sphere": {"scale": 0.28}, "stick": {"radius": 0.14}})
    view.setBackgroundColor("white")


def show_molecule(symbols, X, width=420, height=320, labels=False):
    """Draw one geometry (X in angstrom, shape (N, 3))."""
    if not _have_py3dmol():
        return _mpl_frames(symbols, np.asarray(X)[None], interval=200)
    import py3Dmol
    v = py3Dmol.view(width=width, height=height)
    v.addModel(_xyz_block(symbols, X), "xyz")
    _style(v)
    if labels:
        for i, (s, x) in enumerate(zip(symbols, np.asarray(X))):
            v.addLabel(f"{s}{i}", {"position": {"x": x[0], "y": x[1], "z": x[2]},
                                    "fontSize": 12, "backgroundOpacity": 0.4})
    v.zoomTo()
    return v.show()


def animate_mode(symbols, X0, displacement, amplitude=0.25, nframes=24, width=420, height=320,
                 interval=50):
    """Animate one normal mode.

    displacement : (N, 3) Cartesian displacement pattern of the mode (any scale);
                   it is normalised so the largest atomic displacement equals `amplitude` (angstrom).
    """
    d = np.asarray(displacement, float).reshape(-1, 3)
    d = d / np.abs(d).max() * amplitude
    frames = [np.asarray(X0) + np.sin(2 * np.pi * k / nframes) * d for k in range(nframes)]
    return animate_trajectory(symbols, np.array(frames), stride=1, width=width, height=height,
                              interval=interval)


def animate_trajectory(symbols, traj, stride=1, width=480, height=360, interval=40, center=True):
    """Play an MD trajectory (shape (nframes, N, 3), angstrom) as a looping 3-D movie.

    stride   : show every `stride`-th frame
    interval : milliseconds between frames
    center   : remove the centre-of-geometry drift so the molecule stays in view
    """
    T = np.asarray(traj, float)[::stride]
    if center:
        T = T - T.mean(axis=1, keepdims=True)
    if not _have_py3dmol():
        return _mpl_frames(symbols, T, interval=interval)
    import py3Dmol
    v = py3Dmol.view(width=width, height=height)
    v.addModelsAsFrames("".join(_xyz_block(symbols, X, f"frame {i}") for i, X in enumerate(T)), "xyz")
    _style(v)
    v.zoomTo()
    v.animate({"loop": "forward", "interval": interval})
    return v.show()


def _mpl_frames(symbols, T, interval=40):
    """Fallback animation with matplotlib (no py3Dmol)."""
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML
    fig = plt.figure(figsize=(4.5, 4))
    ax = fig.add_subplot(projection="3d")
    lim = np.abs(T).max() + 0.5
    cols = [CPK.get(s, "#888888") for s in symbols]
    sizes = [600 * RADIUS.get(s, 0.7) ** 2 for s in symbols]

    def draw(i):
        ax.cla()
        X = T[i]
        for a in range(len(X)):
            for b in range(a + 1, len(X)):
                if np.linalg.norm(X[a] - X[b]) < 1.25 * (RADIUS.get(symbols[a], .7) + RADIUS.get(symbols[b], .7)):
                    ax.plot(*zip(X[a], X[b]), color="#555555", lw=3)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=sizes, c=cols, edgecolors="k", depthshade=False)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_axis_off()

    anim = animation.FuncAnimation(fig, draw, frames=len(T), interval=interval)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def trajectory_strip(symbols, traj, times_fs, frames, plane=(2, 0), ax=None):
    """Static film strip of selected frames (for reports and abstracts).

    frames : indices into traj to draw side by side
    plane  : which Cartesian axes to project onto (default z horizontal, x vertical)
    """
    import matplotlib.pyplot as plt
    T = np.asarray(traj, float)
    T = T - T.mean(axis=1, keepdims=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.6 * len(frames), 1.8))
    h, v = plane
    span = np.abs(T[frames][..., h]).max() + 0.6
    vspan = np.abs(T[frames][..., v]).max() + 0.6
    for k, i in enumerate(frames):
        X = T[i]
        off = 2 * span * k
        for a in range(len(X)):
            for b in range(a + 1, len(X)):
                if np.linalg.norm(X[a] - X[b]) < 1.25 * (RADIUS.get(symbols[a], .7) + RADIUS.get(symbols[b], .7)):
                    ax.plot([X[a, h] + off, X[b, h] + off], [X[a, v], X[b, v]], color="#555555", lw=2, zorder=1)
        ax.scatter(X[:, h] + off, X[:, v], s=[350 * RADIUS.get(s, .7) ** 2 for s in symbols],
                   c=[CPK.get(s, "#888") for s in symbols], edgecolors="k", zorder=2)
        ax.text(off, -vspan, f"{times_fs[i]:.0f} fs", ha="center", va="top", fontsize=9)
    ax.set_ylim(-vspan - 0.6, vspan)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax
