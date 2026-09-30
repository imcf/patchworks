"""Review panel for napari: look at the objects most likely to be wrong, fix them.

Open it with ``patchworks review image.zarr`` (or :func:`review_in_napari`).
The panel walks through a queue of objects; for each one the view jumps to
it and shows it with the object it belongs to (outlined) and the objects it
holds, everything else hidden. One key per decision:

====================  ==========================================================
G                     correct as it is
W                     wrong: not a real object
H, then click         belongs to another parent: click the right one (the
                      background means "to none")
J                     join with the suggested object (a split at a tile seam);
                      Shift+J, then click, to join with another one
N / Shift+N           skip / go back
U                     undo the decision about this object
====================  ==========================================================

Outside the queue, click any object to inspect it with its parent and
children (read at full resolution, so thin cilia are easy to hit), type an
id in "Go to #", or hover to see its parent and position in the status bar.
With a position rule, cilia can be coloured by class (apical, basal,
lateral, central), seen from the side (z up), and their class corrected.

Every decision is saved at once, next to the object table in the store, so
the panel can be closed at any time and a later session continues where
this one stopped. All logic lives in :class:`patchworks._review.Review`;
this module is only the view of it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np

from .._review import POSITIONS, QUEUES, Review, _nice

logger = logging.getLogger(__name__)

_POSITION_COLOURS = {
    "apical": "#2ecc71",
    "basal": "#3498db",
    "lateral": "#f39c12",
    "central": "#e84393",
}
_POSITION_CODES = {c: i + 1 for i, c in enumerate(POSITIONS)}

_QUEUE_TITLES = {
    "flagged": "Flagged first",
    "random": "Random sample (error rate)",
    "all": "Everything, by id",
}


def review_in_napari(
    store: Union[str, Path],
    *,
    expect: dict | None = None,
    min_overlap: float | None = None,
    position: dict | None = None,
    show: bool = True,
    viewer: Any = None,
):
    """Open *store* in napari with the review panel docked.

    Parameters
    ----------
    store : str or Path
        OME-ZARR image store whose label images carry object tables
        (the workflow writes them; ``patchworks tables`` adds them).
    expect, min_overlap, position
        Review rules, see :class:`patchworks._review.Review`.
    show : bool
        Start the napari event loop (blocking).
    viewer : napari.Viewer, optional
        Reuse a viewer that already shows the store's layers.

    Returns
    -------
    tuple
        ``(viewer, widget)``.
    """
    from .napari import _require_napari, view_in_napari

    napari = _require_napari()
    rv = Review(
        store, expect=expect, min_overlap=min_overlap, position=position
    )
    if not rv.names:
        raise ValueError(
            f"no label image in {store} has an object table yet: run "
            f"`patchworks tables {store}` first"
            + (f" (stale: {', '.join(rv.stale)})" if rv.stale else "")
        )
    if viewer is None:
        viewer = view_in_napari(store, show=False)
    widget = ReviewWidget(viewer, rv)
    viewer.window.add_dock_widget(widget, name="Review", area="right")
    if show:
        napari.run()
    return viewer, widget


def _widgets():
    from qtpy.QtWidgets import (
        QCheckBox,
        QComboBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    return (
        QCheckBox,
        QComboBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )


_CLASS: Any = None


def _full_resolution_value(layer, event, reach: int = 2) -> int:
    """The label under the cursor, read at full resolution.

    napari reports the value of the pyramid level on screen, where an
    object one voxel wide (a cilium) has vanished. In 2-D, read level 0
    around the clicked voxel instead, and take the nearest object within
    *reach* voxels in the displayed plane, so a thin object is easy to hit.
    """
    if len(event.dims_displayed) != 2:
        value = layer.get_value(
            event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        return int((value[1] if isinstance(value, tuple) else value) or 0)
    data = layer.data[0] if layer.multiscale else layer.data
    point = np.round(np.asarray(layer.world_to_data(event.position))).astype(
        int
    )
    offset = len(event.position) - data.ndim
    shown = [d - offset for d in event.dims_displayed]
    lo = point.copy()
    hi = point + 1
    for d in shown:
        lo[d] -= reach
        hi[d] += reach
    lo = np.clip(lo, 0, data.shape)
    hi = np.clip(hi, 0, data.shape)
    if np.any(hi <= lo):
        return 0
    window = np.asarray(data[tuple(slice(a, b) for a, b in zip(lo, hi))])
    hits = np.argwhere(window > 0)
    if not hits.size:
        return 0
    nearest = hits[np.argmin(np.abs(hits + lo - point).sum(axis=1))]
    return int(window[tuple(nearest)])


def _hex_rgba(colour: str) -> np.ndarray:
    return np.array(
        [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)] + [1.0],
        dtype=np.float32,
    )


def _camera(viewer):
    """napari >= 0.9 moved the camera to ``viewer.scene.camera``."""
    scene = getattr(viewer, "scene", None)
    return getattr(scene, "camera", None) or viewer.camera


def ReviewWidget(viewer: Any, review: Review):
    """The review dock widget (a QWidget) for *review*, driving *viewer*.

    A factory rather than a class so that importing this module needs no Qt.
    """
    global _CLASS
    if _CLASS is None:
        from qtpy.QtWidgets import QWidget

        _CLASS = type("ReviewWidget", (_ReviewPanel, QWidget), {})
    return _CLASS(viewer, review)


class _ReviewPanel:
    """The dock widget's behaviour; mixed into QWidget by :func:`ReviewWidget`.
    See the module docstring for the keys."""

    def __init__(self, viewer: Any, review: Review) -> None:
        (
            QCheckBox,
            QComboBox,
            QFileDialog,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMessageBox,
            QPushButton,
            QVBoxLayout,
            QWidget,
        ) = _widgets()
        super().__init__()
        self.viewer = viewer
        self.rv = review
        self.name = review.names[0]
        self.mode = "flagged"
        self.queue: list[int] = []
        self.current: int | None = None
        self.history: list[int] = []
        self.pending: str | None = None  # "parent" | "merge"
        self._flags: Any = None
        self._saved: dict[str, tuple[Any, int, bool]] = {}
        self._by_parent: dict[str, Any] = {}
        self._by_position: dict[str, Any] = {}
        self._points = None

        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Objects:"))
        self.name_box = QComboBox()
        self.name_box.addItems(review.names)
        row.addWidget(self.name_box)
        row.addWidget(QLabel("Queue:"))
        self.queue_box = QComboBox()
        self.queue_box.addItems([_QUEUE_TITLES[q] for q in QUEUES])
        row.addWidget(self.queue_box)
        layout.addLayout(row)

        self.title = QLabel()
        self.title.setStyleSheet("font-size: 15px; font-weight: bold")
        layout.addWidget(self.title)
        self.reasons = QLabel()
        self.reasons.setWordWrap(True)
        self.reasons.setStyleSheet("color: #ff9f43")
        layout.addWidget(self.reasons)
        self.info = QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        self.prompt = QLabel()
        self.prompt.setWordWrap(True)
        self.prompt.setStyleSheet("color: #54a0ff; font-weight: bold")
        layout.addWidget(self.prompt)

        row = QHBoxLayout()
        row.addWidget(QLabel("Go to #"))
        self.goto_box = QLineEdit()
        self.goto_box.setPlaceholderText("object id, Enter")
        row.addWidget(self.goto_box)
        layout.addLayout(row)

        grid = QGridLayout()
        self.btn_ok = QPushButton("✓ Correct  [G]")
        self.btn_wrong = QPushButton("✗ Not an object  [W]")
        self.btn_parent = QPushButton("Change parent…  [H]")
        self.btn_join = QPushButton("Join with…  [J]")
        self.btn_skip = QPushButton("Skip  [N]")
        self.btn_back = QPushButton("Back  [Shift+N]")
        self.btn_undo = QPushButton("Undo  [U]")
        for i, b in enumerate(
            (
                self.btn_ok,
                self.btn_wrong,
                self.btn_parent,
                self.btn_join,
                self.btn_skip,
                self.btn_back,
                self.btn_undo,
            )
        ):
            grid.addWidget(b, i // 2, i % 2)
        layout.addLayout(grid)
        row = QHBoxLayout()
        row.addWidget(QLabel("Parent image:"))
        self.parent_box = QComboBox()
        row.addWidget(self.parent_box)
        layout.addLayout(row)
        self.position_row = QWidget()
        prow = QHBoxLayout(self.position_row)
        prow.setContentsMargins(0, 0, 0, 0)
        prow.addWidget(QLabel("Position is:"))
        self.position_buttons = {}
        for cls in POSITIONS:
            b = QPushButton(cls)
            b.clicked.connect(lambda _=False, c=cls: self.set_position(c))
            prow.addWidget(b)
            self.position_buttons[cls] = b
        layout.addWidget(self.position_row)

        self.inspect_box = QCheckBox(
            "Click any object to inspect it (with its parent and children)"
        )
        self.inspect_box.setChecked(True)
        layout.addWidget(self.inspect_box)
        self.side_box = QCheckBox(
            "Side view: z against x, z up (apical/basal at a glance)"
        )
        layout.addWidget(self.side_box)
        self.focus_box = QCheckBox("Show only this object and its relatives")
        self.focus_box.setChecked(True)
        layout.addWidget(self.focus_box)
        self.by_parent_box = QCheckBox("Colour these objects by their parent")
        layout.addWidget(self.by_parent_box)
        self.by_position_label = QLabel(
            "Colour by position: "
            + " · ".join(
                f"<span style='color:{_POSITION_COLOURS[c]}'>■ {c}</span>"
                for c in POSITIONS
            )
        )
        self.by_position_box = QCheckBox("Colour these objects by position")
        layout.addWidget(self.by_position_box)
        layout.addWidget(self.by_position_label)

        self.progress = QLabel()
        self.progress.setWordWrap(True)
        layout.addWidget(self.progress)

        out = QGroupBox("Results")
        out_row = QHBoxLayout(out)
        self.format_box = QComboBox()
        self.format_box.addItems(["csv", "xlsx", "parquet"])
        self.btn_export = QPushButton("Export corrected tables…")
        self.btn_write = QPushButton("Write corrected labels")
        out_row.addWidget(self.format_box)
        out_row.addWidget(self.btn_export)
        out_row.addWidget(self.btn_write)
        layout.addWidget(out)
        layout.addStretch(1)

        self._QFileDialog, self._QMessageBox = QFileDialog, QMessageBox
        self.name_box.currentTextChanged.connect(self._set_name)
        self.queue_box.currentIndexChanged.connect(
            lambda i: self._set_mode(QUEUES[i])
        )
        self.btn_ok.clicked.connect(lambda: self.decide("ok"))
        self.btn_wrong.clicked.connect(lambda: self.decide("wrong"))
        self.btn_parent.clicked.connect(lambda: self.start("parent"))
        self.btn_join.clicked.connect(lambda: self.join())
        self.btn_skip.clicked.connect(lambda: self.next())
        self.btn_back.clicked.connect(lambda: self.back())
        self.btn_undo.clicked.connect(lambda: self.undo())
        self.focus_box.toggled.connect(lambda _: self._show())
        self.by_parent_box.toggled.connect(lambda _: self._colour_by_parent())
        self.by_position_box.toggled.connect(
            lambda _: self._colour_by_position()
        )
        self.goto_box.returnPressed.connect(self._goto)
        self.side_box.toggled.connect(self._side_view)
        self.btn_export.clicked.connect(self._export)
        self.btn_write.clicked.connect(self._write_labels)

        keys = {
            "g": lambda v: self.decide("ok"),
            "w": lambda v: self.decide("wrong"),
            "h": lambda v: self.start("parent"),
            "j": lambda v: self.join(),
            "Shift-J": lambda v: self.start("merge"),
            "n": lambda v: self.next(),
            "Shift-N": lambda v: self.back(),
            "u": lambda v: self.undo(),
        }
        for key, fn in keys.items():
            viewer.bind_key(key, fn, overwrite=True)
        viewer.mouse_drag_callbacks.append(self._on_click)
        self._set_name(self.name)

    # -- state -----------------------------------------------------------------

    def _set_name(self, name: str) -> None:
        self._restore()
        self._remove_by_parent()
        self.name = name
        self.current = None
        self.history.clear()
        parents = self.rv.parents[name]
        self.parent_box.clear()
        self.parent_box.addItems(parents)
        self.btn_parent.setEnabled(bool(parents))
        self.parent_box.setEnabled(len(parents) > 1)
        if not parents:
            self.by_parent_box.setChecked(False)
        self.by_parent_box.setEnabled(bool(parents))
        has_position = name in self.rv.position
        self.position_row.setVisible(has_position)
        self._remove_by_position()
        if not has_position:
            self.by_position_box.setChecked(False)
        self.by_position_box.setEnabled(has_position)
        self.by_position_label.setVisible(has_position)
        self._refresh_queue()
        self._update_features()
        self.next(record=False)

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self.current = None
        self._refresh_queue()
        self.next(record=False)

    def _refresh_queue(self) -> None:
        self._flags = self.rv.flag_table(self.name)
        self.queue = self.rv.queue(self.name, self.mode)
        self._update_points()
        self._update_progress()

    # -- navigation ------------------------------------------------------------

    def next(self, record: bool = True) -> None:
        """Go to the next object of the queue still to review; a skipped
        object moves to the end, so N cycles through the queue."""
        self.pending = None
        if record and self.current is not None:
            self.history.append(self.current)
            if self.current in self.queue:
                self.queue.remove(self.current)
                self.queue.append(self.current)
        upcoming = [x for x in self.queue if x != self.current]
        self.go(upcoming[0] if upcoming else None)

    def back(self) -> None:
        self.pending = None
        if self.history:
            self.go(self.history.pop())

    def go(self, label: int | None) -> None:
        """Show object *label* (None: the queue is done)."""
        self.current = label
        self.prompt.setText("")
        if label is None:
            self.title.setText(
                f"{_nice(self.name)}: nothing left in this queue"
            )
            self.reasons.setText("")
            self.info.setText(
                "Pick another queue, or export the corrected tables."
            )
            self._restore()
            return
        self._describe()
        self._show()

    def _describe(self) -> None:
        label = self.current
        eff = self.rv.effective(self.name)
        decided = self.rv.decisions[self.name].get(label)
        status = f"  —  reviewed: {decided['action']}" if decided else ""
        left = len([x for x in self.queue if x != label])
        self.title.setText(
            f"{_nice(self.name)} #{label}{status}   ({left} more in queue)"
        )
        self.reasons.setText(
            "\n".join(f"• {r}" for r in self.rv.reasons(self.name, label))
        )
        partner = self._partner(label)
        self.btn_join.setText(
            f"Join with #{partner}  [J]" if partner else "Join with…  [J]"
        )
        if label not in eff.index:  # rejected, or merged away
            self.info.setText("(no longer in the corrected table)")
            return
        row = eff.loc[label]
        lines = []
        if "area_um3" in row:
            lines.append(
                f"volume {row['area_um3']:.3g} µm³ ({int(row['area_voxels'])} voxels)"
            )
        else:
            lines.append(f"{int(row['area_voxels'])} voxels")
        for p in self.rv.parents[self.name]:
            pid = int(row[f"{p}_id"])
            ov = row.get(f"{p}_overlap")
            where = (
                f"in {_nice(p)} #{pid}"
                + (f" ({ov:.0%} inside)" if ov == ov and pid else "")
                if pid
                else f"in no {_nice(p)}"
            )
            lines.append(where)
        for c in self.rv.children[self.name]:
            lines.append(f"{int(row[f'n_{c}'])} {_nice(c)}")
        for unit in ("um", "voxels"):
            if (
                f"length_{unit}" in row
                and row[f"length_{unit}"] == row[f"length_{unit}"]
            ):
                lines.append(
                    f"length {row[f'length_{unit}']:.3g} "
                    + ("µm" if unit == "um" else "voxels")
                )
        if "position" in row:
            where = str(row["position"])
            if row.get("angle_to_axis_deg") == row.get("angle_to_axis_deg"):
                where += f", {row['angle_to_axis_deg']:.0f}° to the apical axis"
            lines.append(where)
        self.info.setText(" · ".join(lines))

    # -- decisions -------------------------------------------------------------

    def _record(self, action: str, **kw) -> None:
        if self.current is None:
            return
        try:
            self.rv.decide(
                self.name, self.current, action, queue=self.mode, **kw
            )
        except PermissionError as exc:
            self._QMessageBox.critical(self, "Cannot save", str(exc))
            return
        except (KeyError, ValueError) as exc:
            self.prompt.setText(str(exc))
            return
        done = self.current
        self.pending = None
        self._refresh_queue()
        self._update_by_parent()
        self._update_by_position()
        self._update_features()
        self.history.append(done)
        self.current = None
        upcoming = [x for x in self.queue if x != done]
        self.go(upcoming[0] if upcoming else None)

    def decide(self, action: str) -> None:
        """``ok`` or ``wrong`` for the current object."""
        self._record(action)

    def start(self, what: str) -> None:
        """Wait for a click: the right parent, or the object to join."""
        if self.current is None:
            return
        if what == "parent" and not self.rv.parents[self.name]:
            return
        if self.pending == what:  # pressing again cancels
            self.pending = None
            self.prompt.setText("")
            return
        self.pending = what
        target = (
            _nice(self.parent_box.currentText())
            if what == "parent"
            else f"{_nice(self.name)} object"
        )
        self.prompt.setText(
            f"Click the right {target}"
            + (" (or the background for none)" if what == "parent" else "")
            + ". Press the key again to cancel."
        )

    def _partner(self, label: int | None) -> int | None:
        """The object a seam flag suggests joining *label* with."""
        if label is None or label not in self._flags.index:
            return None
        partner = int(self._flags.at[label, "partner"])
        return partner if partner > 0 else None

    def join(self) -> None:
        partner = self._partner(self.current)
        if partner:
            self._record("merge", into=partner)
        else:
            self.start("merge")

    def undo(self) -> None:
        if self.current is None:
            return
        self.rv.undo(self.name, self.current)
        self._refresh_queue()
        self._update_by_parent()
        self._update_by_position()
        self._update_features()
        self._describe()
        self._show()

    def _on_click(self, viewer, event):
        """A click (not a drag): picks the object a pending decision waits
        for, or else (inspect mode) shows the object under the cursor."""
        if self.pending is None and not self.inspect_box.isChecked():
            return
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged = True
            yield
        if dragged:
            return
        if self.pending is None:
            hit = self._object_at(viewer, event)
            if hit is not None:
                self.inspect(*hit)
            return
        layer_name = (
            self.parent_box.currentText()
            if self.pending == "parent"
            else self.name
        )
        if layer_name not in viewer.layers:
            return
        layer = viewer.layers[layer_name]
        self.pick(
            layer.get_value(
                event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
        )

    def _object_at(self, viewer, event) -> tuple[str, int] | None:
        """The topmost reviewed object under the cursor: a cilium over its
        cell, a nucleus over its cell, else the cell."""
        for layer in reversed(list(viewer.layers)):
            if layer.name not in self.rv.tables or not layer.visible:
                continue
            value = _full_resolution_value(layer, event)
            if value and int(value) in self.rv.tables[layer.name].index:
                return layer.name, int(value)
        return None

    def inspect(self, name: str, label: int) -> None:
        """Show object *label* of *name*, with its parent and children --
        whether or not it is in the queue."""
        if name != self.name:
            self.name_box.blockSignals(True)
            self.name_box.setCurrentText(name)
            self.name_box.blockSignals(False)
            self._set_name(name)
        if self.current is not None and self.current != label:
            self.history.append(self.current)
        self.go(label)

    def _goto(self) -> None:
        text = self.goto_box.text().strip().lstrip("#")
        if text.isdigit() and int(text) in self.rv.tables[self.name].index:
            self.inspect(self.name, int(text))
            self.goto_box.clear()
        else:
            self.prompt.setText(f"{_nice(self.name)} has no object {text!r}")

    def set_position(self, cls: str) -> None:
        """Correct the position class of the current object."""
        self._record("position", position=cls)

    def pick(self, value: Any) -> None:
        """Apply the pending decision with the object id *value* clicked."""
        if isinstance(value, tuple):  # multiscale: (level, value)
            value = value[1]
        value = int(value or 0)
        if self.pending == "parent":
            self._record(
                "parent",
                parent=self.parent_box.currentText(),
                parent_id=value,
            )
        elif self.pending == "merge":
            if not value or value == self.current:
                self.prompt.setText("Click another object to join with.")
                return
            self._record("merge", into=value)

    # -- the view --------------------------------------------------------------

    def _layer(self, name: str):
        return self.viewer.layers[name] if name in self.viewer.layers else None

    def _save_layer(self, layer) -> None:
        if layer.name not in self._saved:
            self._saved[layer.name] = (
                layer.colormap,
                layer.contour,
                layer.visible,
            )

    def _restore(self) -> None:
        for name, (cmap, contour, visible) in self._saved.items():
            layer = self._layer(name)
            if layer is not None:
                layer.colormap = cmap
                layer.contour = contour
                layer.visible = visible
        self._saved.clear()

    def _only(self, layer, ids, *, outline: bool = False) -> None:
        from napari.utils.colormaps import DirectLabelColormap

        self._save_layer(layer)
        base = self._saved[layer.name][0]
        ids = [int(i) for i in ids if i]
        colors = base.map(np.asarray(ids)) if ids else []
        clear = np.zeros(4, dtype=np.float32)
        layer.colormap = DirectLabelColormap(
            color_dict={
                **{i: c for i, c in zip(ids, colors)},
                0: clear,
                None: clear,
            }
        )
        layer.contour = 2 if outline else 0
        layer.visible = True

    def _show(self) -> None:
        """Frame the current object; in focus mode, hide everything else."""
        label = self.current
        layer = self._layer(self.name)
        if label is None or layer is None:
            return
        table = self.rv.tables[self.name]
        if label not in table.index:
            return
        row = table.loc[label]
        axes = self.rv.meta[self.name].get("axes") or "zyx"[-layer.ndim :]
        centre = [float(row[f"centroid_{ax}"]) for ax in axes]
        lo = np.array([row[f"bbox_min_{ax}"] for ax in axes], float)
        hi = np.array([row[f"bbox_max_{ax}"] for ax in axes], float) + 1
        # Frame the object with its parent: a cilium alone is a few voxels,
        # and where it sits in its cell is what the eye needs.
        eff = self.rv.effective(self.name)
        for p in self.rv.parents[self.name]:
            pid = int(eff.loc[label, f"{p}_id"]) if label in eff.index else 0
            ptable = self.rv.tables[p]
            if pid in ptable.index:
                prow = ptable.loc[pid]
                lo = np.minimum(lo, [prow[f"bbox_min_{ax}"] for ax in axes])
                hi = np.maximum(hi, [prow[f"bbox_max_{ax}"] + 1 for ax in axes])
        self._frame(layer, centre, lo, hi)
        self._restore()
        if not self.focus_box.isChecked():
            return
        eff = self.rv.effective(self.name)
        self._only(layer, [label])
        if label in eff.index:
            for p in self.rv.parents[self.name]:
                player = self._layer(p)
                if player is not None:
                    self._only(
                        player, [eff.loc[label, f"{p}_id"]], outline=True
                    )
            for c in self.rv.children[self.name]:
                clayer = self._layer(c)
                if clayer is not None:
                    ce = self.rv.effective(c)
                    self._only(clayer, ce.index[ce[f"{self.name}_id"] == label])
        by = self._by_parent.get(self.name)
        if by is not None:
            self._save_layer(by)
            by.visible = False

    def _frame(self, layer, centre, lo, hi) -> None:
        """Put the slice through *centre*, and fit the box *lo*..*hi* in view."""
        world = np.asarray(layer.data_to_world(centre), float)
        middle = np.asarray(layer.data_to_world((lo + hi - 1) / 2), float)
        dims = self.viewer.dims
        offset = dims.ndim - world.size
        for i in range(world.size):
            if offset + i not in dims.displayed:
                dims.set_point(offset + i, world[i])
        shown = [d - offset for d in dims.displayed if d >= offset]
        camera = _camera(self.viewer)
        camera.center = tuple(middle[shown])
        scale = np.asarray(layer.scale[-world.size :], float)
        # At least 32 voxels across, so a lone small object keeps context.
        extent = float(
            max(((hi - lo) * scale)[shown].max(), 32 * scale[shown].min())
        )
        try:
            canvas = min(self.viewer.window._qt_viewer.canvas.size)
        except Exception:
            canvas = 600
        if extent > 0:
            camera.zoom = canvas / (extent * 1.3)

    def _update_points(self) -> None:
        """Mark the flagged objects still open: visible in 3D too, where
        small objects vanish at napari's coarse 3D level."""
        layer = self._layer(self.name)
        if layer is None:
            return
        table = self.rv.tables[self.name]
        axes = self.rv.meta[self.name].get("axes") or "zyx"[-layer.ndim :]
        open_ids = [
            int(f)
            for f in self._flags.index
            if int(f) not in self.rv.decisions[self.name]
        ]
        pts = (
            table.loc[open_ids, [f"centroid_{ax}" for ax in axes]].to_numpy()
            if open_ids
            else np.empty((0, len(axes)))
        )
        name = f"{self.name}: to review"
        if self._points is not None and self._points.name in self.viewer.layers:
            if self._points.name == name:
                self._points.data = pts
                return
            self.viewer.layers.remove(self._points)
        self._points = self.viewer.add_points(
            pts,
            name=name,
            scale=layer.scale,
            translate=layer.translate,
            units=layer.units,
            size=6,
            face_color="transparent",
            border_color="#ff9f43",
            border_width=0.2,
            opacity=0.8,
        )
        self.viewer.layers.selection.active = layer

    def _update_progress(self) -> None:
        s = self.rv.summary(self.name)
        text = (
            f"{s['flagged_open']} of {s['flagged']} flagged still open · "
            f"{s['reviewed']} reviewed ({s['ok']} correct, {s['wrong']} "
            f"rejected, {s['fixed']} fixed"
            + (
                f", {s['position_fixed']} positions corrected"
                if s["position_fixed"]
                else ""
            )
            + ")"
        )
        if s["sample"]:
            lo, hi = s["error_ci"]
            text += (
                f"\nError rate {s['error_rate']:.1%} (95% CI {lo:.1%}–"
                f"{hi:.1%}) from {s['sample']} random objects"
            )
        elif self.mode == "random":
            text += "\nReview in this order: the error rate appears as you go."
        self.progress.setText(text)

    # -- colour by parent --------------------------------------------------------

    def _lut(self) -> np.ndarray:
        parent = self.parent_box.currentText()
        table = self.rv.tables[self.name]
        eff = self.rv.effective(self.name)
        lut = np.zeros(int(table.index.max()) + 1, dtype=np.int32)
        roots = self.rv.roots(self.name)
        for label in table.index:
            root = roots.get(int(label), int(label))
            if root in eff.index:
                lut[label] = eff.loc[root, f"{parent}_id"]
        return lut

    def _colour_by_parent(self) -> None:
        if not self.by_parent_box.isChecked():
            self._remove_by_parent()
            return
        layer = self._layer(self.name)
        parent = self._layer(self.parent_box.currentText())
        if layer is None or parent is None:
            return
        lut = self._lut()
        levels = layer.data if layer.multiscale else [layer.data]

        def mapped(level):
            import dask.array as da

            arr = da.asarray(level)
            return arr.map_blocks(
                lambda b: lut[np.clip(b, 0, lut.size - 1)], dtype=np.int32
            )

        by = self.viewer.add_labels(
            [mapped(lv) for lv in levels]
            if layer.multiscale
            else mapped(levels[0]),
            name=f"{self.name} by {self.parent_box.currentText()}",
            multiscale=layer.multiscale,
            scale=layer.scale,
            translate=layer.translate,
            units=layer.units,
            colormap=parent.colormap,
        )
        by.metadata["lut"] = lut
        self._by_parent[self.name] = by

    def _update_by_parent(self) -> None:
        by = self._by_parent.get(self.name)
        if by is not None and by.name in self.viewer.layers:
            by.metadata["lut"][...] = self._lut()
            by.refresh()

    def _remove_by_parent(self) -> None:
        by = self._by_parent.pop(self.name, None)
        if by is not None and by.name in self.viewer.layers:
            self.viewer.layers.remove(by)

    def _side_view(self, on: bool) -> None:
        """Show z against x through the object (z up), or the usual y/x."""
        dims = self.viewer.dims
        n = dims.ndim
        if n < 3:
            return
        rest = list(range(n - 3))
        z, y, x = n - 3, n - 2, n - 1
        dims.order = tuple(rest + ([y, z, x] if on else [z, y, x]))
        camera = _camera(self.viewer)
        try:
            camera.orientation2d = ("up", "right") if on else ("down", "right")
        except Exception:  # pragma: no cover - napari < 0.5
            pass
        self._show()

    # -- colour by position, hover -------------------------------------------

    def _position_lut(self) -> np.ndarray:
        table = self.rv.tables[self.name]
        eff = self.rv.effective(self.name)
        roots = self.rv.roots(self.name)
        codes = eff["position"].map(_POSITION_CODES).fillna(0).astype(np.int32)
        lut = np.zeros(int(table.index.max()) + 1, dtype=np.int32)
        for label in table.index:
            root = roots.get(int(label), int(label))
            if root in codes.index:
                lut[label] = codes[root]
        return lut

    def _colour_by_position(self) -> None:
        from napari.utils.colormaps import DirectLabelColormap

        if not self.by_position_box.isChecked():
            self._remove_by_position()
            return
        layer = self._layer(self.name)
        if layer is None or self.name not in self.rv.position:
            return
        lut = self._position_lut()
        levels = layer.data if layer.multiscale else [layer.data]

        def mapped(level):
            import dask.array as da

            return da.asarray(level).map_blocks(
                lambda b: lut[np.clip(b, 0, lut.size - 1)], dtype=np.int32
            )

        colours: dict[int | None, np.ndarray] = {
            code: _hex_rgba(_POSITION_COLOURS[c])
            for c, code in _POSITION_CODES.items()
        }
        grey = np.array([0.6, 0.6, 0.6, 1.0], dtype=np.float32)
        palette: dict[int | None, np.ndarray] = {
            **colours,
            0: np.zeros(4, dtype=np.float32),
            None: grey,
        }
        by = self.viewer.add_labels(
            [mapped(lv) for lv in levels]
            if layer.multiscale
            else mapped(levels[0]),
            name=f"{self.name} by position",
            multiscale=layer.multiscale,
            scale=layer.scale,
            translate=layer.translate,
            units=layer.units,
            colormap=DirectLabelColormap(color_dict=palette),
        )
        by.metadata["lut"] = lut
        self._by_position[self.name] = by

    def _update_by_position(self) -> None:
        by = self._by_position.get(self.name)
        if by is not None and by.name in self.viewer.layers:
            by.metadata["lut"][...] = self._position_lut()
            by.refresh()

    def _remove_by_position(self) -> None:
        by = self._by_position.pop(self.name, None)
        if by is not None and by.name in self.viewer.layers:
            self.viewer.layers.remove(by)

    def _update_features(self) -> None:
        """What napari's status bar shows when hovering an object: its
        parent, children and position (the layer's per-label features)."""
        for name in self.rv.names:
            layer = self._layer(name)
            if layer is None:
                continue
            eff = self.rv.effective(name)
            cols = [f"{p}_id" for p in self.rv.parents[name]]
            cols += [f"n_{c}" for c in self.rv.children[name]]
            cols += [c for c in ("position", "qc") if c in eff]
            features = (
                eff[cols]
                .reset_index()
                .rename(columns={eff.index.name or "label": "index"})
            )
            try:
                layer.features = features
            except Exception:  # pragma: no cover - depends on napari
                logger.debug(
                    "could not set features on %s", name, exc_info=True
                )

    # -- results ---------------------------------------------------------------

    def _export(self) -> None:
        folder = self._QFileDialog.getExistingDirectory(
            self, "Export corrected tables to"
        )
        if folder:
            paths = self.rv.export(folder, self.format_box.currentText())
            self.prompt.setText(f"Wrote {len(paths)} table(s) to {folder}")

    def _write_labels(self) -> None:
        from napari.qt.threading import thread_worker

        name = self.name
        self.btn_write.setEnabled(False)
        self.prompt.setText(f"Writing {name}_reviewed…")

        @thread_worker
        def work():
            return self.rv.write_reviewed_labels(name)

        worker = work()
        worker.returned.connect(
            lambda out: (
                self.prompt.setText(f"Wrote labels/{out} (reopen to view it)"),
                self.btn_write.setEnabled(True),
            )
        )
        worker.errored.connect(
            lambda exc: (
                self._QMessageBox.critical(self, "Write failed", str(exc)),
                self.btn_write.setEnabled(True),
            )
        )
        worker.start()
        self._worker = worker
