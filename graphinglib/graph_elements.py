from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any, Literal, Optional, Protocol, Sequence, cast, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from numpy.typing import ArrayLike

from .legend_artists import VerticalLineCollection

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@runtime_checkable
class Plottable(Protocol):
    """
    Dummy class for a general plottable object.

    .. attention:: Not to be used directly.

    """

    label: str | None
    handle: Any

    def _plot_element(
        self, axes: plt.Axes, z_order: int, *, cycle_color: str | None = None
    ) -> None: ...

    def copy_with(self, **kwargs) -> Self:
        """
        Returns a deep copy of the Plottable with specified attributes overridden. This is useful when multiple
        properties need to be changed in copies of Plottable objects, as it allows to modify the attributes in a single
        call.

        Parameters
        ----------
        **kwargs
            Properties to override in the copied Plottable. The keys should be property names to modify and the values
            are the new values for those properties.

        Returns
        -------
        Self
            A new instance with the specified attributes overridden.

        Examples
        --------
        Copy an existing Curve and change the color and line_style at the same time::

            curve = Curve(x_data, y_data, color='blue')
            new_curve = curve.copy_with(color='red', line_style='dashed')
        """
        properties = [
            attr
            for attr in dir(self.__class__)
            if isinstance(getattr(self.__class__, attr, None), property)
        ]
        properties = list(
            filter(lambda x: x[0] != "_", properties)
        )  # filter out hidden properties
        print(properties)
        new_copy = deepcopy(self)
        for key, value in kwargs.items():
            if hasattr(new_copy, key):
                setattr(new_copy, key, value)
            else:
                close_match = get_close_matches(key, properties, n=1, cutoff=0.6)
                if close_match:
                    raise AttributeError(
                        f"{self.__class__.__name__} has no attribute '{key}'. "
                        f"Did you mean '{close_match[0]}'?"
                    )
                else:
                    raise AttributeError(
                        f"{self.__class__.__name__} has no attribute '{key}'."
                    )
        return new_copy

    def __deepcopy__(self, memo: dict) -> Self:
        """
        Creates a deep copy of the Plottable instance, intentionally excluding the 'handle' attribute from the copy.
        This avoids issues when copying a Plottable that has been previously drawn and stored.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        excluded_attrs = ["handle"]
        for property_, value in self.__dict__.items():
            if property_ not in excluded_attrs:
                result.__dict__[property_] = deepcopy(value, memo)
        for attr in excluded_attrs:
            if hasattr(self, attr):
                setattr(result, attr, None)
        return result

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        pass


class GraphingException(Exception):
    """
    General exception raised for the GraphingLib modules.
    """

    pass


class Hlines(Plottable):
    """
    This class implements simple horizontal lines.

    Parameters
    ----------
    y : ArrayLike
        Vertical positions at which the lines should be plotted.
    x_min : ArrayLike, optional
        Horizontal start position of the lines. Each lines can have a different start.
        If not specified, lines will span the entire axes. Defaults to ``None``.
    x_max : ArrayLike, optional
        Horizontal end position of the lines. Each lines can habe a different end.
        If not specified, lines will span the entire axes. Defaults to ``None``.
    label : str, optional
        Label to be displayed in the legend.
    colors : list[str]
        Colors to use for the lines. One color for every line or a color
        per line can be specified.
        Default depends on the ``figure_style`` configuration.
    line_widths : list[float]
        Line widths to use for the lines. One width for every line or a width
        per line can be specified.
        Default depends on the ``figure_style`` configuration.
    line_styles : list[str]
        Line styles to use for the lines. One style for every line or a style
        per line can be specified.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the lines.
        Default depends on the ``figure_style`` configuration.
    """

    def __init__(
        self,
        y: ArrayLike,
        x_min: Optional[ArrayLike] = None,
        x_max: Optional[ArrayLike] = None,
        label: Optional[str] = None,
        colors: list[str] | str = "default",
        line_widths: list[float] | float | Literal["default"] = "default",
        line_styles: list[str] | str = "default",
        alpha: float | Literal["default"] = "default",
    ) -> None:
        if isinstance(y, (list, np.ndarray)):
            self._y = np.asarray(y)
        else:
            self._y = y
        if isinstance(x_min, (list, np.ndarray)):
            self._x_min = np.asarray(x_min)
        else:
            self._x_min = x_min
        if isinstance(x_max, (list, np.ndarray)):
            self._x_max = np.asarray(x_max)
        else:
            self._x_max = x_max
        if (self._x_min is None) ^ (self._x_max is None):
            raise GraphingException(
                "Either both x_min and x_max are specified or none of them"
            )
        self._label = label
        self._colors = colors
        self._line_widths = line_widths
        self._line_styles = line_styles
        self._alpha = alpha
        if isinstance(self._y, (int, float)) and isinstance(
            self._colors, (list, np.ndarray)
        ):
            if len(self._colors) > 1:
                raise GraphingException(
                    "There can't be multiple colors for a single line!"
                )
        if isinstance(self._y, (int, float)) and isinstance(
            self._line_styles, (list, np.ndarray)
        ):
            if len(self._line_styles) > 1:
                raise GraphingException(
                    "There can't be multiple line styles for a single line!"
                )
        if isinstance(self._y, (int, float)) and isinstance(
            self._line_widths, (list, np.ndarray)
        ):
            if len(self._line_widths) > 1:
                raise GraphingException(
                    "There can't be multiple line widths for a single line!"
                )
        if isinstance(self._y, (list, np.ndarray)):
            if isinstance(self._colors, list) and len(self._y) != len(self._colors):
                raise GraphingException(
                    "There must be the same number of colors and lines!"
                )
            if isinstance(self._line_styles, list) and len(self._y) != len(
                self._line_styles
            ):
                raise GraphingException(
                    "There must be the same number of line styles and lines!"
                )

            if isinstance(self._line_widths, list) and len(self._y) != len(
                self._line_widths
            ):
                raise GraphingException(
                    "There must be the same number of line widths and lines!"
                )

    @property
    def y(self) -> ArrayLike:
        return self._y

    @y.setter
    def y(self, y: ArrayLike) -> None:
        self._y = y

    @property
    def x_min(self) -> ArrayLike | None:
        return self._x_min

    @x_min.setter
    def x_min(self, x_min: Optional[ArrayLike]) -> None:
        self._x_min = x_min

    @property
    def x_max(self) -> ArrayLike | None:
        return self._x_max

    @x_max.setter
    def x_max(self, x_max: Optional[ArrayLike]) -> None:
        self._x_max = x_max

    @property
    def label(self) -> Optional[str]:
        return self._label

    @label.setter
    def label(self, label: Optional[str]) -> None:
        self._label = label

    @property
    def colors(self) -> list[str] | str:
        return self._colors

    @colors.setter
    def colors(self, colors: list[str] | str) -> None:
        self._colors = colors

    @property
    def line_widths(self) -> list[float] | float | Literal["default"]:
        return self._line_widths

    @line_widths.setter
    def line_widths(
        self, line_widths: list[float] | float | Literal["default"]
    ) -> None:
        self._line_widths = line_widths

    @property
    def line_styles(self) -> list[str] | str:
        return self._line_styles

    @line_styles.setter
    def line_styles(self, line_styles: list[str] | str) -> None:
        self._line_styles = line_styles

    @property
    def alpha(self) -> float | Literal["default"]:
        return self._alpha

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.graph_elements.Hlines` object.
        """
        return deepcopy(self)

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        colors = cast(str | list[str] | np.ndarray, self._colors)
        if colors == "default":
            raise ValueError(
                "Hlines colors is still set to 'default' when plotting; "
                "resolve style parameters before rendering."
            )
        if isinstance(colors, np.ndarray):
            colors = colors.tolist()
        colors = cast(str | list[str], colors)

        line_styles = cast(str | list[str] | np.ndarray, self._line_styles)
        if line_styles == "default":
            raise ValueError(
                "Hlines line_styles is still set to 'default' when plotting; "
                "resolve style parameters before rendering."
            )
        if isinstance(line_styles, np.ndarray):
            line_styles = line_styles.tolist()
        line_styles = cast(str | list[str], line_styles)

        line_widths = cast(float | list[float] | np.ndarray, self._line_widths)
        if line_widths == "default":
            raise ValueError(
                "Hlines line_widths is still set to 'default' when plotting; "
                "resolve style parameters before rendering."
            )
        if isinstance(line_widths, np.ndarray):
            line_widths = line_widths.tolist()
        line_widths = cast(float | list[float], line_widths)

        alpha_value = self._alpha
        if alpha_value == "default":
            raise ValueError(
                "Hlines alpha is still set to 'default' when plotting; "
                "resolve style parameters before rendering."
            )
        alpha: float = float(alpha_value)

        y_values: list[float] = [
            float(v) for v in np.atleast_1d(np.asarray(self._y, dtype=float))
        ]
        y_len = len(y_values)
        x_min_values: list[float] | None = (
            [float(v) for v in np.atleast_1d(np.asarray(self._x_min, dtype=float))]
            if self._x_min is not None
            else None
        )
        x_max_values: list[float] | None = (
            [float(v) for v in np.atleast_1d(np.asarray(self._x_max, dtype=float))]
            if self._x_max is not None
            else None
        )

        def _select(value: list[Any] | tuple[Any, ...], idx: int):
            return value[idx]

        def _pick(seq: Sequence[float], idx: int) -> float:
            return seq[idx] if len(seq) > 1 else seq[0]

        if isinstance(self._y, (list, np.ndarray)) and len(self._y) > 1:
            if self._x_max is not None and self._x_min is not None:
                assert x_min_values is not None
                assert x_max_values is not None
                y_seq = cast(Sequence[float], y_values)
                xmin_seq = cast(Sequence[float], x_min_values)
                xmax_seq = cast(Sequence[float], x_max_values)
                for i, y_val in enumerate(y_seq):
                    axes.hlines(
                        y_val,
                        _pick(xmin_seq, i),
                        _pick(xmax_seq, i),
                        zorder=z_order,
                        colors=_select(colors, i)
                        if isinstance(colors, (list, tuple))
                        else colors,
                        linestyles=_select(line_styles, i)
                        if isinstance(line_styles, (list, tuple))
                        else line_styles,
                        linewidths=_select(line_widths, i)
                        if isinstance(line_widths, (list, tuple))
                        else line_widths,
                        alpha=alpha,
                    )
                self.handle = LineCollection(
                    [[(0, 0)]] * (y_len if y_len <= 3 else 3),
                    colors=colors,
                    linestyles=line_styles,
                    linewidths=line_widths,
                    alpha=alpha,
                )
            else:
                for i, y_val in enumerate(y_values):
                    axes.axhline(
                        y_val,
                        zorder=z_order,
                        color=_select(colors, i)
                        if isinstance(colors, (list, tuple))
                        else colors,
                        linestyle=_select(line_styles, i)
                        if isinstance(line_styles, (list, tuple))
                        else line_styles,
                        linewidth=_select(line_widths, i)
                        if isinstance(line_widths, (list, tuple))
                        else line_widths,
                        alpha=alpha,
                    )
                self.handle = LineCollection(
                    [[(0, 0)]] * (y_len if y_len <= 3 else 3),
                    colors=colors,
                    linestyles=line_styles,
                    linewidths=line_widths,
                    alpha=alpha,
                )
        else:
            if self._x_max is not None and self._x_min is not None:
                assert x_min_values is not None
                assert x_max_values is not None
                y_seq = cast(Sequence[float], y_values)
                xmin_seq = cast(Sequence[float], x_min_values)
                xmax_seq = cast(Sequence[float], x_max_values)
                color_single = colors[0] if isinstance(colors, list) else colors
                linestyle_single = (
                    line_styles[0] if isinstance(line_styles, list) else line_styles
                )
                linewidth_single = (
                    line_widths[0] if isinstance(line_widths, list) else line_widths
                )
                axes.hlines(
                    y_seq[0],
                    xmin_seq[0],
                    xmax_seq[0],
                    zorder=z_order,
                    colors=color_single,
                    linestyles=linestyle_single,
                    linewidths=linewidth_single,
                    alpha=alpha,
                )
            else:
                color_single = colors[0] if isinstance(colors, list) else colors
                linestyle_single = (
                    line_styles[0] if isinstance(line_styles, list) else line_styles
                )
                linewidth_single = (
                    line_widths[0] if isinstance(line_widths, list) else line_widths
                )
                axes.axhline(
                    y_values[0],
                    zorder=z_order,
                    color=color_single,
                    linestyle=linestyle_single,
                    linewidth=linewidth_single,
                    alpha=alpha,
                )
            if isinstance(self._y, (int, float)):
                self.handle = LineCollection(
                    [[(0, 0)]] * 1,
                    colors=colors,
                    linestyles=line_styles,
                    linewidths=line_widths,
                    alpha=alpha,
                )
            else:
                self.handle = LineCollection(
                    [[(0, 0)]] * (y_len if y_len <= 3 else 3),
                    colors=colors,
                    linestyles=line_styles,
                    linewidths=line_widths,
                    alpha=alpha,
                )


class Vlines(Plottable):
    """
    This class implements simple vertical lines.

    Parameters
    ----------
    x : ArrayLike
        Horizontal positions at which the lines should be plotted.
    y_min : ArrayLike, optional
        Vertical start position of the lines. Each line can have a different start.
        If not specified, lines will span the entire axes. Defaults to ``None``.
    y_max : ArrayLike, optional
        Vertical end position of the lines. Each line can habe a different end.
        If not specified, lines will span the entire axes. Defaults to ``None``.
    label : str, optional
        Label to be displayed in the legend.
    colors : list[str]
        Colors to use for the lines. One color for every line or a color
        per line can be specified.
        Default depends on the ``figure_style`` configuration.
    line_widths : list[float]
        Line widths to use for the lines. One width for every line or a width
        per line can be specified.
        Default depends on the ``figure_style`` configuration.
    line_styles : list[str]
        Line styles to use for the lines. One style for every line or a style
        per line can be specified.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the lines.
        Default depends on the ``figure_style`` configuration.
    """

    def __init__(
        self,
        x: ArrayLike,
        y_min: Optional[ArrayLike] = None,
        y_max: Optional[ArrayLike] = None,
        label: Optional[str] = None,
        colors: list[str] | str = "default",
        line_widths: list[float] | float | Literal["default"] = "default",
        line_styles: list[str] | str = "default",
        alpha: float | Literal["default"] = "default",
    ) -> None:
        if isinstance(x, (list, np.ndarray)):
            self._x = np.asarray(x)
        else:
            self._x = x
        if isinstance(y_min, (list, np.ndarray)):
            self._y_min = np.asarray(y_min)
        else:
            self._y_min = y_min
        if isinstance(y_max, (list, np.ndarray)):
            self._y_max = np.asarray(y_max)
        else:
            self._y_max = y_max
        self._label = label
        self._colors = colors
        self._line_styles = line_styles
        self._line_widths = line_widths
        self._alpha = alpha
        if isinstance(self._x, (int, float)) and isinstance(
            self._colors, (list, np.ndarray)
        ):
            if len(self._colors) > 1:
                raise GraphingException(
                    "There can't be multiple colors for a single line!"
                )
        if isinstance(self._x, (int, float)) and isinstance(
            self._line_styles, (list, np.ndarray)
        ):
            if len(self._line_styles) > 1:
                raise GraphingException(
                    "There can't be multiple line styles for a single line!"
                )
        if isinstance(self._x, (int, float)) and isinstance(
            self._line_widths, (list, np.ndarray)
        ):
            if len(self._line_widths) > 1:
                raise GraphingException(
                    "There can't be multiple line widths for a single line!"
                )
        if isinstance(self._x, (list, np.ndarray)):
            if isinstance(self._colors, list) and len(self._x) != len(self._colors):
                raise GraphingException(
                    "There must be the same number of colors and lines!"
                )
            if isinstance(self._line_styles, list) and len(self._x) != len(
                self._line_styles
            ):
                raise GraphingException(
                    "There must be the same number of line styles and lines!"
                )

            if isinstance(self._line_widths, list) and len(self._x) != len(
                self._line_widths
            ):
                raise GraphingException(
                    "There must be the same number of line widths and lines!"
                )

    @property
    def x(self) -> ArrayLike:
        return self._x

    @x.setter
    def x(self, x: ArrayLike) -> None:
        self._x = x

    @property
    def y_min(self) -> ArrayLike | None:
        return self._y_min

    @y_min.setter
    def y_min(self, y_min: Optional[ArrayLike]) -> None:
        self._y_min = y_min

    @property
    def y_max(self) -> ArrayLike | None:
        return self._y_max

    @y_max.setter
    def y_max(self, y_max: Optional[ArrayLike]) -> None:
        self._y_max = y_max

    @property
    def label(self) -> Optional[str]:
        return self._label

    @label.setter
    def label(self, label: Optional[str]) -> None:
        self._label = label

    @property
    def colors(self) -> list[str] | str:
        return self._colors

    @colors.setter
    def colors(self, colors: list[str] | str) -> None:
        self._colors = colors

    @property
    def line_widths(self) -> list[float] | float | Literal["default"]:
        return self._line_widths

    @line_widths.setter
    def line_widths(
        self, line_widths: list[float] | float | Literal["default"]
    ) -> None:
        self._line_widths = line_widths

    @property
    def line_styles(self) -> list[str] | str:
        return self._line_styles

    @line_styles.setter
    def line_styles(self, line_styles: list[str] | str) -> None:
        self._line_styles = line_styles

    @property
    def alpha(self) -> float | Literal["default"]:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float | Literal["default"]) -> None:
        self._alpha = alpha

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.graph_elements.Vlines` object.
        """
        return deepcopy(self)

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        colors = cast(str | list[str] | np.ndarray, self._colors)
        if colors == "default":
            raise ValueError(
                "Vlines colors is still set to 'default' when plotting; "
                "resolve style parameters before rendering."
            )
        if isinstance(colors, np.ndarray):
            colors = colors.tolist()
        colors = cast(str | list[str], colors)

        line_styles = cast(str | list[str] | np.ndarray, self._line_styles)
        if line_styles == "default":
            raise ValueError(
                "Vlines line_styles is still set to 'default' when plotting; "
                "resolve style parameters before rendering."
            )
        if isinstance(line_styles, np.ndarray):
            line_styles = line_styles.tolist()
        line_styles = cast(str | list[str], line_styles)

        line_widths = cast(float | list[float] | np.ndarray, self._line_widths)
        if line_widths == "default":
            raise ValueError(
                "Vlines line_widths is still set to 'default' when plotting; "
                "resolve style parameters before rendering."
            )
        if isinstance(line_widths, np.ndarray):
            line_widths = line_widths.tolist()
        line_widths = cast(float | list[float], line_widths)

        alpha_value = self._alpha
        if alpha_value == "default":
            raise ValueError(
                "Vlines alpha is still set to 'default' when plotting; "
                "resolve style parameters before rendering."
            )
        alpha: float = float(alpha_value)

        x_values: list[float] = [
            float(v) for v in np.atleast_1d(np.asarray(self._x, dtype=float))
        ]
        x_len = len(x_values)
        y_min_values: list[float] | None = (
            [float(v) for v in np.atleast_1d(np.asarray(self._y_min, dtype=float))]
            if self._y_min is not None
            else None
        )
        y_max_values: list[float] | None = (
            [float(v) for v in np.atleast_1d(np.asarray(self._y_max, dtype=float))]
            if self._y_max is not None
            else None
        )

        def _select(value: list[Any] | tuple[Any, ...], idx: int):
            return value[idx]

        def _pick(seq: Sequence[float], idx: int) -> float:
            return seq[idx] if len(seq) > 1 else seq[0]

        if isinstance(self._x, (list, np.ndarray)) and len(self._x) > 1:
            if self._y_min is not None and self._y_max is not None:
                assert y_min_values is not None
                assert y_max_values is not None
                x_seq = cast(Sequence[float], x_values)
                ymin_seq = cast(Sequence[float], y_min_values)
                ymax_seq = cast(Sequence[float], y_max_values)
                for i, x_val in enumerate(x_seq):
                    axes.vlines(
                        x_val,
                        _pick(ymin_seq, i),
                        _pick(ymax_seq, i),
                        zorder=z_order,
                        colors=_select(colors, i)
                        if isinstance(colors, (list, tuple))
                        else colors,
                        linestyles=_select(line_styles, i)
                        if isinstance(line_styles, (list, tuple))
                        else line_styles,
                        linewidths=_select(line_widths, i)
                        if isinstance(line_widths, (list, tuple))
                        else line_widths,
                        alpha=alpha,
                    )
                self.handle = VerticalLineCollection(
                    [[(0, 0)]] * (x_len if x_len <= 4 else 4),
                    colors=colors,
                    linestyles=line_styles,
                    linewidths=line_widths,
                    alpha=alpha,
                )
            else:
                for i, x_val in enumerate(x_values):
                    axes.axvline(
                        x_val,
                        zorder=z_order,
                        color=_select(colors, i)
                        if isinstance(colors, (list, tuple))
                        else colors,
                        linestyle=_select(line_styles, i)
                        if isinstance(line_styles, (list, tuple))
                        else line_styles,
                        linewidth=_select(line_widths, i)
                        if isinstance(line_widths, (list, tuple))
                        else line_widths,
                        alpha=alpha,
                    )
                self.handle = VerticalLineCollection(
                    [[(0, 0)]] * (x_len if x_len <= 4 else 4),
                    colors=colors,
                    linestyles=line_styles,
                    linewidths=line_widths,
                    alpha=alpha,
                )
        else:
            if self._y_min is not None and self._y_max is not None:
                assert y_min_values is not None
                assert y_max_values is not None
                x_seq = cast(Sequence[float], x_values)
                ymin_seq = cast(Sequence[float], y_min_values)
                ymax_seq = cast(Sequence[float], y_max_values)
                color_single = colors[0] if isinstance(colors, list) else colors
                linestyle_single = (
                    line_styles[0] if isinstance(line_styles, list) else line_styles
                )
                linewidth_single = (
                    line_widths[0] if isinstance(line_widths, list) else line_widths
                )
                axes.vlines(
                    x_seq[0],
                    ymin_seq[0],
                    ymax_seq[0],
                    zorder=z_order,
                    colors=color_single,
                    linestyles=linestyle_single,
                    linewidths=linewidth_single,
                    alpha=alpha,
                )
            else:
                color_single = colors[0] if isinstance(colors, list) else colors
                linestyle_single = (
                    line_styles[0] if isinstance(line_styles, list) else line_styles
                )
                linewidth_single = (
                    line_widths[0] if isinstance(line_widths, list) else line_widths
                )
                axes.axvline(
                    x_values[0],
                    zorder=z_order,
                    color=color_single,
                    linestyle=linestyle_single,
                    linewidth=linewidth_single,
                    alpha=alpha,
                )
            if isinstance(self._x, (int, float)):
                self.handle = VerticalLineCollection(
                    [[(0, 0)]] * 1,
                    colors=colors,
                    linestyles=line_styles,
                    linewidths=line_widths,
                    alpha=alpha,
                )
            else:
                self.handle = VerticalLineCollection(
                    [[(0, 0)]] * (x_len if x_len <= 4 else 4),
                    colors=colors,
                    linestyles=line_styles,
                    linewidths=line_widths,
                    alpha=alpha,
                )


class Point(Plottable):
    """
    This class implements a point object.

    The :class:`~graphinglib.graph_elements.Point`
    object can be used to show important coordinates in a plot
    or add a label to some point.

    Parameters
    ----------
    x, y : float
        The x and y coordinates of the :class:`~graphinglib.graph_elements.Point`.
    label : str, optional
        Label to be attached to the :class:`~graphinglib.graph_elements.Point`.
    face_color : str or None
        Face color of the marker.
        Default depends on the ``figure_style`` configuration.
    edge_color : str or None
        Edge color of the marker.
        Default depends on the ``figure_style`` configuration.
    marker_size : float
        Size of the marker.
        Default depends on the ``figure_style`` configuration.
    marker_style : str
        Style of the marker.
        Default depends on the ``figure_style`` configuration.
    edge_width : float
        Edge width of the marker.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the point.
        Default depends on the ``figure_style`` configuration.
    font_size : float
        Font size for the text attached to the marker.
        Default depends on the ``figure_style`` configuration.
    text_color : str
        Color of the text attached to the marker.
        "same as point" uses the color of the point (prioritize edge color, then face color). Default depends on the ``figure_style`` configuration.
    h_align, v_align : str
        Horizontal and vertical alignment of the text attached
        to the :class:`~graphinglib.graph_elements.Point`.
        Defaults to bottom left.
    """

    def __init__(
        self,
        x: float,
        y: float,
        label: Optional[str] = None,
        face_color: Optional[str] = "default",
        edge_color: Optional[str] = "default",
        marker_size: float | Literal["default"] = "default",
        marker_style: str = "default",
        edge_width: float | Literal["default"] = "default",
        alpha: float | Literal["default"] = "default",
        font_size: int | Literal["same as figure"] = "same as figure",
        text_color: str = "default",
        h_align: str = "left",
        v_align: str = "bottom",
    ) -> None:
        """
        This class implements a point object.

        The point object can be used to show important coordinates in a plot
        or add a label to some point.

        Parameters
        ----------
        x, y : float
            The x and y coordinates of the :class:`~graphinglib.graph_elements.Point`.
        label : str, optional
            Label to be attached to the :class:`~graphinglib.graph_elements.Point`.
        face_color : str or None
            Face color of the marker.
            Default depends on the ``figure_style`` configuration.
        edge_color : str or None
            Edge color of the marker.
            Default depends on the ``figure_style`` configuration.
        marker_size : float
            Size of the marker.
            Default depends on the ``figure_style`` configuration.
        marker_style : str
            Style of the marker.
            Default depends on the ``figure_style`` configuration.
        edge_width : float
            Edge width of the marker.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the point.
            Default depends on the ``figure_style`` configuration.
        font_size : float
            Font size for the text attached to the marker.
            Default depends on the ``figure_style`` configuration.
        text_color : str
            Color of the text attached to the marker.
            "same as point" uses the color of the point (prioritize edge color, then face color). Default depends on the ``figure_style`` configuration.
        h_align, v_align : str
            Horizontal and vertical alignment of the text attached
            to the :class:`~graphinglib.graph_elements.Point`.
            Defaults to bottom left.
        """
        if not isinstance(x, int | float) or not isinstance(y, int | float):
            raise GraphingException(
                "The x and y coordinates for a point must be a single number each!"
            )
        else:
            self._x = x
            self._y = y
        self._label = label
        self._face_color = face_color
        self._edge_color = edge_color
        self._marker_size = marker_size
        self._marker_style = marker_style
        self._edge_width = edge_width
        self._alpha = alpha
        self._font_size = font_size
        self._text_color = text_color
        self._h_align = h_align
        self._v_align = v_align
        self._show_coordinates: bool = False

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        self._x = x

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        self._y = y

    @property
    def label(self) -> Optional[str]:
        return self._label

    @label.setter
    def label(self, label: Optional[str]) -> None:
        self._label = label

    @property
    def face_color(self) -> str | None:
        return self._face_color

    @face_color.setter
    def face_color(self, face_color: str) -> None:
        self._face_color = face_color

    @property
    def edge_color(self) -> str | None:
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: str) -> None:
        self._edge_color = edge_color

    @property
    def marker_size(self) -> float | Literal["default"]:
        return self._marker_size

    @marker_size.setter
    def marker_size(self, marker_size: float | Literal["default"]) -> None:
        self._marker_size = marker_size

    @property
    def marker_style(self) -> str:
        return self._marker_style

    @marker_style.setter
    def marker_style(self, marker_style: str) -> None:
        self._marker_style = marker_style

    @property
    def edge_width(self) -> float | Literal["default"]:
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width: float | Literal["default"]) -> None:
        self._edge_width = edge_width

    @property
    def alpha(self) -> float | Literal["default"]:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float | Literal["default"]) -> None:
        self._alpha = alpha

    @property
    def font_size(self) -> float | Literal["same as figure"]:
        return self._font_size

    @font_size.setter
    def font_size(self, font_size: float | Literal["same as figure"]) -> None:
        self._font_size = font_size

    @property
    def text_color(self) -> str:
        return self._text_color

    @text_color.setter
    def text_color(self, text_color: str) -> None:
        self._text_color = text_color

    @property
    def h_align(self) -> str:
        return self._h_align

    @h_align.setter
    def h_align(self, h_align: str) -> None:
        self._h_align = h_align

    @property
    def v_align(self) -> str:
        return self._v_align

    @v_align.setter
    def v_align(self, v_align: str) -> None:
        self._v_align = v_align

    @property
    def show_coordinates(self) -> bool:
        return self._show_coordinates

    @show_coordinates.setter
    def show_coordinates(self, show_coordinates: bool) -> None:
        self._show_coordinates = show_coordinates

    @property
    def coordinates(self) -> tuple[float, float]:
        return (self._x, self._y)

    @coordinates.setter
    def coordinates(self, coordinates: tuple[float, float]) -> None:
        self._x, self._y = coordinates

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.graph_elements.Point` object.
        """
        return deepcopy(self)

    def add_coordinates(self) -> None:
        """
        Displays the coordinates of the :class:`~graphinglib.graph_elements.Point` next to it.
        """
        self._show_coordinates = True

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        if self._face_color is None and self._edge_color is None:
            raise GraphingException(
                "Both the face color and edge color of the point can't be None. Set at least one of them."
            )
        if isinstance(self._marker_size, str):
            raise ValueError("Point marker_size is still set to 'default' when plotting.")
        if isinstance(self._edge_width, str):
            raise ValueError("Point edge_width is still set to 'default' when plotting.")
        if isinstance(self._alpha, str):
            raise ValueError("Point alpha is still set to 'default' when plotting.")

        size = self._font_size if self._font_size != "same as figure" else None
        prefix = " " if self._h_align == "left" else ""
        postfix = " " if self._h_align == "right" else ""
        point_label = (
            prefix + self._label + postfix if self._label is not None and not self._show_coordinates else None
        )
        face_color = self._face_color if self._face_color is not None else "none"
        edge_color = self._edge_color if self._edge_color is not None else "none"

        scatter_params: dict[str, Any] = {
            "c": face_color,
            "edgecolors": edge_color,
            "s": float(self._marker_size),
            "marker": self._marker_style,
            "linewidths": float(self._edge_width),
            "alpha": float(self._alpha),
        }
        axes.scatter(self._x, self._y, zorder=z_order, **scatter_params)
        # get text color. if _text_color is "same as point", use the color of the point (prioritize edge color, then face color)
        if self._text_color == "same as point":
            if self._edge_color is not None:
                text_color = self._edge_color
            else:
                text_color = self._face_color
        else:
            text_color = self._text_color
        annotate_params: dict[str, Any] = {
            "color": text_color,
            "horizontalalignment": self._h_align,
            "verticalalignment": self._v_align,
        }
        if size is not None:
            annotate_params["fontsize"] = size
        if point_label is not None:
            axes.annotate(point_label, (self._x, self._y), zorder=z_order, **annotate_params)
        if self._show_coordinates:
            prefix = " " if self._h_align == "left" else ""
            postfix = " " if self._h_align == "right" else ""
            if self._label is not None:
                point_label = (
                    prefix
                    + self._label
                    + " : "
                    + f"({self._x:.3f}, {self._y:.3f})"
                    + postfix
                )
            else:
                point_label = prefix + f"({self._x:.3f}, {self._y:.3f})" + postfix
            if self._text_color == "same as point":
                if self._edge_color is not None:
                    text_color = self._edge_color
                else:
                    text_color = self._face_color
            else:
                text_color = self._text_color
            params: dict[str, Any] = {
                "color": text_color if text_color is not None else "black",
                "horizontalalignment": self._h_align,
                "verticalalignment": self._v_align,
            }
            if size is not None:
                params["fontsize"] = size
            axes.annotate(
                point_label,
                (self._x, self._y),
                zorder=z_order,
                **params,
            )


@dataclass
class Text(Plottable):
    """
    This class allows text to be plotted.

    It is also possible to attach an arrow to the :class:`~graphinglib.graph_elements.Text`
    with the method :py:meth:`~graphinglib.graph_elements.Text.attach_arrow`
    to point at something of interest in the plot.

    Parameters
    ----------
    x, y : float
        The x and y coordinates at which to plot the :class:`~graphinglib.graph_elements.Text`.
    text : str
        The text to be plotted.
    color : str
        Color of the text.
        Default depends on the ``figure_style`` configuration.
    font_size : float
        Font size of the text.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the text.
        Default depends on the ``figure_style`` configuration.
    h_align, v_align : str
        Horizontal and vertical alignment of the text.
        Default depends on the ``figure_style`` configuration.
    rotation : float
        Rotation angle of the text in degrees.
        Defaults to 0.
    highlight_color : str, optional
        Color of the background highlight box behind the text.
        If specified, a box will be drawn behind the text.
        Default is ``None`` (no highlight).
    highlight_alpha : float, optional
        Opacity of the highlight box.
        Defaults to 1.0.
    highlight_padding : float, optional
        Padding around the text for the highlight box. A value of 0 means no padding.
        Defaults to 0.1.
    """

    _x: float
    _y: float
    _text: str
    _color: str = "default"
    _font_size: float | Literal["same as figure"] = "same as figure"
    _alpha: float | Literal["default"] = "default"
    _h_align: str = "default"
    _v_align: str = "default"
    _rotation: float = 0.0
    _highlight_color: Optional[str] = None
    _highlight_alpha: float = 1.0
    _highlight_padding: float = 0.1
    _arrow_pointing_to: Optional[tuple[float, float]] = field(default=None, init=False)

    def __init__(
        self,
        x: float,
        y: float,
        text: str,
        color: str = "default",
        font_size: float | Literal["same as figure"] = "same as figure",
        alpha: float | Literal["default"] = "default",
        h_align: str = "default",
        v_align: str = "default",
        rotation: float = 0.0,
        highlight_color: Optional[str] = None,
        highlight_alpha: float = 1.0,
        highlight_padding: float = 0.1,
    ) -> None:
        """
        This class allows text to be plotted.

        It is also possible to attach an arrow to the :class:`~graphinglib.graph_elements.Text`
        with the method :py:meth:`~graphinglib.graph_elements.Text.attach_arrow`
        to point at something of interest in the plot.

        Parameters
        ----------
        x, y : float
            The x and y coordinates at which to plot the :class:`~graphinglib.graph_elements.Text`.
        text : str
            The text to be plotted.
        color : str
            Color of the text.
            Default depends on the ``figure_style`` configuration.
        font_size : float
            Font size of the text.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the text.
            Default depends on the ``figure_style`` configuration.
        h_align, v_align : str
            Horizontal and vertical alignment of the text.
            Default depends on the ``figure_style`` configuration.
        rotation : float
            Rotation angle of the text in degrees.
            Defaults to 0.
        highlight_color : str, optional
            Color of the background highlight box behind the text.
            If specified, a box will be drawn behind the text.
            Default is ``None`` (no highlight).
        highlight_alpha : float, optional
            Opacity of the highlight box.
            Defaults to 1.0.
        highlight_padding : float, optional
            Padding around the text for the highlight box. A value of 0 means no padding.
            Defaults to 0.1.
        """
        self._x = x
        self._y = y
        self._text = text
        self._color = color
        self._font_size = font_size
        self._alpha = alpha
        self._h_align = h_align
        self._v_align = v_align
        self._rotation = rotation
        self._highlight_color = highlight_color
        self._highlight_alpha = highlight_alpha
        self._highlight_padding = highlight_padding
        self._arrow_pointing_to = None

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        self._x = x

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        self._y = y

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, text: str) -> None:
        self._text = text

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, color: str) -> None:
        self._color = color

    @property
    def font_size(self) -> float | Literal["same as figure"]:
        return self._font_size

    @font_size.setter
    def font_size(self, font_size: float | Literal["same as figure"]) -> None:
        self._font_size = font_size

    @property
    def alpha(self) -> float | Literal["default"]:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float | Literal["default"]) -> None:
        self._alpha = alpha

    @property
    def h_align(self) -> str:
        return self._h_align

    @h_align.setter
    def h_align(self, h_align: str) -> None:
        self._h_align = h_align

    @property
    def v_align(self) -> str:
        return self._v_align

    @v_align.setter
    def v_align(self, v_align: str) -> None:
        self._v_align = v_align

    @property
    def rotation(self) -> float:
        return self._rotation

    @rotation.setter
    def rotation(self, rotation: float) -> None:
        self._rotation = rotation

    @property
    def highlight_color(self) -> Optional[str]:
        return self._highlight_color

    @highlight_color.setter
    def highlight_color(self, highlight_color: Optional[str]) -> None:
        self._highlight_color = highlight_color

    @property
    def highlight_alpha(self) -> float:
        return self._highlight_alpha

    @highlight_alpha.setter
    def highlight_alpha(self, highlight_alpha: float) -> None:
        self._highlight_alpha = highlight_alpha

    @property
    def highlight_padding(self) -> float:
        return self._highlight_padding

    @highlight_padding.setter
    def highlight_padding(self, highlight_padding: float) -> None:
        self._highlight_padding = highlight_padding

    @property
    def arrow_pointing_to(self) -> Optional[tuple[float, float]]:
        return self._arrow_pointing_to

    @arrow_pointing_to.setter
    def arrow_pointing_to(self, arrow_pointing_to: Optional[tuple[float, float]]) -> None:
        self._arrow_pointing_to = arrow_pointing_to

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.graph_elements.Text` object.
        """
        return deepcopy(self)

    def add_arrow(
        self,
        points_to: tuple[float, float],
        width: Optional[float] = None,
        shrink: Optional[float] = None,
        head_width: Optional[float] = None,
        head_length: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> None:
        """
        Adds an arrow pointing from the :class:`~graphinglib.graph_elements.Text`
        to a specified point.

        Parameters
        ----------
        points_to: tuple[float, float]
            Coordinates at which to point.
        width : float, optional
            Arrow width.
        shrink : float, optional
            Fraction of the total length of the arrow to shrink from both ends.
            A value of 0.5 means the arrow is no longer visible.
        head_width : float, optional
            Width of the head of the arrow.
        head_length : float, optional
            Length of the head of the arrow.
        alpha : float, optional
            Opacity of the arrow.
        """
        self._arrow_pointing_to = points_to
        self._arrow_properties = {}
        if width is not None:
            self._arrow_properties["width"] = width
        if shrink is not None:
            self._arrow_properties["shrink"] = shrink
        if head_width is not None:
            self._arrow_properties["headwidth"] = head_width
        if head_length is not None:
            self._arrow_properties["headlength"] = head_length
        if alpha is not None:
            self._arrow_properties["alpha"] = alpha

    def _plot_element(
        self, axes: plt.Axes, z_order: int, **kwargs
    ) -> None:
        """
        Plots the element in the specified target, which can be either an
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_ or a
        `Figure <https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html>`.
        Figure type has been omitted in the signature to keep Plottable consistent, but it
        is still supported.
        """
        if self._color == "default":
            raise ValueError("Text color is still set to 'default' when plotting.")
        if isinstance(self._alpha, str):
            raise ValueError("Text alpha is still set to 'default' when plotting.")
        if self._h_align == "default" or self._v_align == "default":
            raise ValueError("Text alignment is still set to 'default' when plotting.")

        size = self._font_size if self._font_size != "same as figure" else None
        params: dict[str, Any] = {
            "color": self._color,
            "alpha": float(self._alpha),
            "horizontalalignment": self._h_align,
            "verticalalignment": self._v_align,
            "rotation": self._rotation,
        }
        if size is not None:
            params["fontsize"] = size

        # Add highlight/background box if highlight_color is specified
        if self._highlight_color is not None:
            bbox_dict = {
                "boxstyle": f"square,pad={self._highlight_padding}",
                "facecolor": self._highlight_color,
                "edgecolor": "none",
                "alpha": self._highlight_alpha,
            }
            params["bbox"] = bbox_dict

        axes.text(
            self._x,
            self._y,
            self._text,
            zorder=z_order,
            **params,
        )
        if self._arrow_pointing_to is not None:
            self._arrow_properties["color"] = self._color
            arrow_params: dict[str, Any] = {
                "color": self._color,
                "horizontalalignment": self._h_align,
                "verticalalignment": self._v_align,
                "arrowprops": self._arrow_properties,
            }
            if size is not None:
                arrow_params["fontsize"] = size
            axes.annotate(
                self._text,
                self._arrow_pointing_to,
                xytext=(self._x, self._y),
                zorder=z_order,
                **arrow_params,
            )


@dataclass
class Table(Plottable):
    """
    This class allows to plot a table inside a Figure or MultiFigure.

    The Table object can be used to add raw data to a figure or add supplementary
    information like output parameters for a fit or anyother operation.

    Parameters
    ----------
    cell_text : list[str]
        Text or data to be displayed in the table. The shape of the provided data
        determines the number of columns and rows.
    cell_colors : ArrayLike or str, optional
        Colors to apply to the cells' background. Must be a list of colors the same
        shape as the cells.
        Default depends on the ``figure_style`` configuration.
    cell_align : str
        Alignment of the cells' text. Must be one of the following:
        {'left', 'center', 'right'}. Default depends on the ``figure_style`` configuration.
    col_labels : list[str], optional
        List of labels for the rows of the table. If none are specified, no row labels are displayed.
    col_widths : list[float], optional
        Widths to apply to the columns. Must be a list the same length as the number of columns.
    col_align : str
        Alignment of the column labels' text. Must be one of the following:
        {'left', 'center', 'right'}. Default depends on the ``figure_style`` configuration.
    col_colors : ArrayLike or str, optional
        Colors to apply to the column labels' background. Must be a list of colors the same
        length as the number of columns.
        Default depends on the ``figure_style`` configuration.
    row_labels : list[str], optional
        List of labels for the rows of the table. If none are specified, no row labels are displayed.
    row_align : str
        Alignment of the row labels' text. Must be one of the following:
        {'left', 'center', 'right'}. Default depends on the ``figure_style`` configuration.
    row_colors : ArrayLike or str, optional
        Colors to apply to the row labels' background. Must be a list of colors the same
        length as the number of rows.
        Default depends on the ``figure_style`` configuration.
    edge_width : float or str, optional
        Width of the table's edges.
        Default depends on the ``figure_style`` configuration.
    edge_color : str, optional
        Color of the table's edges.
        Default depends on the ``figure_style`` configuration.
    text_color : str, optional
        Color of the text in the table.
        Default depends on the ``figure_style`` configuration.
    scaling : tuple[float], optional
        Horizontal and vertical scaling factors to apply to the table.
        Defaults to ``(1, 1.5)``.
    location : str
        Position of the table inside the axes. Must be one of the following:
        {'best', 'bottom', 'bottom left', 'bottom right', 'center', 'center left', 'center right',
        'left', 'lower center', 'lower left', 'lower right', 'right', 'top', 'top left', 'top right',
        'upper center', 'upper left', 'upper right'}
        Defaults to ``"best"``.
    """

    def __init__(
        self,
        cell_text: list[str],
        cell_colors: ArrayLike | str = "default",
        cell_align: str = "default",
        col_labels: Optional[list[str]] = None,
        col_widths: Optional[list[float]] = None,
        col_align: str = "default",
        col_colors: ArrayLike | str = "default",
        row_labels: Optional[list[str]] = None,
        row_align: str = "default",
        row_colors: ArrayLike | str = "default",
        edge_width: float | Literal["default"] = "default",
        edge_color: str = "default",
        text_color: str = "default",
        scaling: tuple[float, float] = (1.0, 1.5),
        location: str = "best",
    ) -> None:
        """
        This class allows to plot a table inside a Figure or MultiFigure.

        The Table object can be used to add raw data to a figure or add supplementary
        information like output parameters for a fit or anyother operation.

        Parameters
        ----------
        cell_text : list[str]
            Text or data to be displayed in the table. The shape of the provided data
            determines the number of columns and rows.
        cell_colors : ArrayLike or str, optional
            Colors to apply to the cells' background. Must be a list of colors the same
            shape as the cells.
            Default depends on the ``figure_style`` configuration.
        cell_align : str
            Alignment of the cells' text. Must be one of the following:
            {'left', 'center', 'right'}. Default depends on the ``figure_style`` configuration.
        col_labels : list[str], optional
            List of labels for the rows of the table. If none are specified, no row labels are displayed.
        col_widths : list[float], optional
            Widths to apply to the columns. Must be a list the same length as the number of columns.
        col_align : str
            Alignment of the column labels' text. Must be one of the following:
            {'left', 'center', 'right'}. Default depends on the ``figure_style`` configuration.
        col_colors : ArrayLike or str, optional
            Colors to apply to the column labels' background. Must be a list of colors the same
            length as the number of columns.
            Default depends on the ``figure_style`` configuration
        row_labels : list[str], optional
            List of labels for the rows of the table. If none are specified, no row labels are displayed.
        row_align : str
            Alignment of the row labels' text. Must be one of the following:
            {'left', 'center', 'right'}. Default depends on the ``figure_style`` configuration.
        row_colors : ArrayLike or str, optional
            Colors to apply to the row labels' background. Must be a list of colors the same
            length as the number of rows.
            Default depends on the ``figure_style`` configuration.
        edge_width : float or str, optional
            Width of the table's edges.
            Default depends on the ``figure_style`` configuration.
        edge_color : str, optional
            Color of the table's edges.
            Default depends on the ``figure_style`` configuration.
        text_color : str, optional
            Color of the text within the table.
            Default depends on the ``figure_style`` configuration.
        scaling : tuple[float], optional
            Horizontal and vertical scaling factors to apply to the table.
            Defaults to ``(1, 1.5)``.
        location : str
            Position of the table inside the axes. Must be one of the following:
            {'best', 'bottom', 'bottom left', 'bottom right', 'center', 'center left', 'center right',
            'left', 'lower center', 'lower left', 'lower right', 'right', 'top', 'top left', 'top right',
            'upper center', 'upper left', 'upper right'}
            Defaults to ``"best"``.
        """
        self._cell_text = cell_text
        self._cell_colors = cell_colors
        self._cell_align = cell_align
        self._col_labels = col_labels
        self._col_widths = col_widths
        self._col_align = col_align
        self._col_colors = col_colors
        self._row_labels = row_labels
        self._row_align = row_align
        self._row_colors = row_colors
        self._edge_width = edge_width
        self._edge_color = edge_color
        self._text_color = text_color
        self._scaling = scaling
        self._location = location

    @property
    def cell_text(self) -> list[str] | None:
        return self._cell_text

    @cell_text.setter
    def cell_text(self, cell_text: list[str] | None) -> None:
        self._cell_text = cell_text

    @property
    def cell_colors(self) -> ArrayLike | str | None:
        return self._cell_colors

    @cell_colors.setter
    def cell_colors(self, cell_colors: list | None) -> None:
        self._cell_colors = cell_colors

    @property
    def cell_align(self) -> str:
        return self._cell_align

    @cell_align.setter
    def cell_align(self, cell_align: str) -> None:
        self._cell_align = cell_align

    @property
    def col_labels(self) -> list[str] | None:
        return self._col_labels

    @col_labels.setter
    def col_labels(self, col_labels: list[str] | None) -> None:
        self._col_labels = col_labels

    @property
    def col_widths(self) -> list[float] | None:
        return self._col_widths

    @col_widths.setter
    def col_widths(self, col_widths: list[float] | None) -> None:
        self._col_widths = col_widths

    @property
    def col_align(self) -> str:
        return self._col_align

    @col_align.setter
    def col_align(self, col_align: str) -> None:
        self._col_align = col_align

    @property
    def col_colors(self) -> ArrayLike | str | None:
        return self._col_colors

    @col_colors.setter
    def col_colors(self, col_colors: list | None) -> None:
        self._col_colors = col_colors

    @property
    def row_labels(self) -> list[str] | None:
        return self._row_labels

    @row_labels.setter
    def row_labels(self, row_labels: list[str] | None) -> None:
        self._row_labels = row_labels

    @property
    def row_align(self) -> str:
        return self._row_align

    @row_align.setter
    def row_align(self, row_align: str) -> None:
        self._row_align = row_align

    @property
    def row_colors(self) -> ArrayLike | str | None:
        return self._row_colors

    @row_colors.setter
    def row_colors(self, row_colors: list | None) -> None:
        self._row_colors = row_colors

    @property
    def edge_width(self) -> float | Literal["default"]:
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width: float | Literal["default"]) -> None:
        self._edge_width = edge_width
        for (i, j), cell in self.handle.get_celld().items():
            cell.set_linewidth(self._edge_width)

    @property
    def edge_color(self) -> str:
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: str) -> None:
        self._edge_color = edge_color
        for (i, j), cell in self.handle.get_celld().items():
            cell.set_edgecolor(self._edge_color)

    @property
    def text_color(self) -> str:
        return self._text_color

    @text_color.setter
    def text_color(self, text_color: str) -> None:
        self._text_color = text_color
        for (i, j), cell in self.handle.get_celld().items():
            cell.set_text_props(color=self._text_color)

    @property
    def scaling(self) -> tuple[float, float]:
        return self._scaling

    @scaling.setter
    def scaling(self, scaling: tuple[float, float]) -> None:
        self._scaling = scaling

    @property
    def location(self) -> str:
        return self._location

    @location.setter
    def location(self, location: str) -> None:
        self._location = location

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.graph_elements.Table` object.
        """
        return deepcopy(self)

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        if self._cell_text is None:
            raise ValueError("Table cell_text must be provided before plotting.")
        cell_array = np.atleast_2d(np.asarray(self._cell_text))
        cell_text: list[list[str]] = [
            [str(val) for val in row] for row in cell_array.tolist()
        ]
        col_labels = self._col_labels
        col_widths = self._col_widths
        col_colors = self._col_colors
        row_labels = self._row_labels
        row_colors = self._row_colors
        if isinstance(self._edge_width, str):
            raise ValueError("Table edge_width is still set to 'default' when plotting.")

        if self._cell_align == "default" or self._col_align == "default" or self._row_align == "default":
            raise ValueError("Table alignment parameters must be resolved before plotting.")
        if self._cell_colors == "default" or col_colors == "default" or row_colors == "default":
            raise ValueError("Table colors must be resolved before plotting.")
        if self._location == "default":
            raise ValueError("Table location is still set to 'default' when plotting.")

        cell_align = self._cell_align
        col_align = self._col_align
        row_align = self._row_align

        def _align(val: str) -> Literal["left", "center", "right"]:
            if val in ("left", "center", "right"):
                return cast(Literal["left", "center", "right"], val)
            raise ValueError(f"Invalid alignment '{val}' for table.")

        cell_loc = _align(cell_align)
        col_loc = _align(col_align)
        row_loc = _align(row_align)

        # Set colors to correct shape if they are strings or arrays
        cell_colours: Sequence[Sequence[str]] | None
        if self._cell_colors is None or self._cell_colors == "default":
            cell_colours = None
        elif isinstance(self._cell_colors, str):
            cell_colours = [[self._cell_colors] * len(cell_text[0])] * len(cell_text)
        else:
            cell_colours = np.atleast_2d(np.asarray(self._cell_colors)).tolist()

        if col_colors == "default":
            col_colors = None
        col_colours: Sequence[str] | None
        if isinstance(col_colors, np.ndarray):
            col_colours = [str(c) for c in np.atleast_1d(col_colors).tolist()]
        elif isinstance(col_colors, list):
            col_colours = [str(c) for c in col_colors]
        elif isinstance(col_colors, str):
            col_colours = [col_colors] * len(cell_text[0])
        else:
            col_colours = None

        if row_colors == "default":
            row_colors = None
        row_colours: Sequence[str] | None
        if isinstance(row_colors, np.ndarray):
            row_colours = [str(c) for c in np.atleast_1d(row_colors).tolist()]
        elif isinstance(row_colors, list):
            row_colours = [str(c) for c in row_colors]
        elif isinstance(row_colors, str):
            row_colours = [row_colors] * len(cell_text)
        else:
            row_colours = None

        if self._location == "default":
            raise ValueError("Table location is still set to 'default' when plotting.")

        loc_value = cast(str, self._location)

        self.handle = axes.table(
            cellText=cell_text,
            cellColours=cell_colours,
            colLabels=col_labels,
            colWidths=col_widths,
            colColours=col_colours,
            rowLabels=row_labels,
            rowColours=row_colours,
            cellLoc=cell_loc,
            colLoc=col_loc,
            rowLoc=row_loc,
            loc=loc_value,
            zorder=z_order,
        )
        self.handle.auto_set_font_size(False)
        if len(self._scaling) < 2:
            raise ValueError("Table scaling must be a tuple of length 2.")
        self.handle.scale(self._scaling[0], self._scaling[1])
        for (i, j), cell in self.handle.get_celld().items():
            cell.set_text_props(color=self._text_color)
            cell.set_edgecolor(self._edge_color)
            cell.set_linewidth(self._edge_width)


class PlottableAxMethod(Plottable):
    """
    This experimental class allows to call any matplotlib Axes method as a plottable element in a
    :class:`~graphinglib.smart_figure.SmartFigure`. This object can be used to create plot types that have not yet been
    implemented in GraphingLib.

    This class only works with Axes methods that create plottable elements (e.g., ``bar`` or ``pcolormesh``).
    Methods that modify axes properties (e.g., ``set_facecolor``, ``set_title``) are not supported.

    Parameters
    ----------
    meth : str
        Name of the matplotlib Axes method to call. The method will be called as ``axes.meth(*args, **kwargs)``. For
        example, this can be "pcolormesh" or "bar".

        .. warning::
            The provided matplotlib Axes method must accept a ``zorder`` keyword argument to be compatible with this
            class. If not, an exception will be raised when attempting to plot the element.
    *args
        Positional arguments to pass to ``axes.meth``.
    **kwargs
        Keyword arguments to pass to ``axes.meth``.
    """

    def __init__(self, meth: str, *args, **kwargs) -> None:
        self.meth = meth
        self.args = args
        self.kwargs = kwargs

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        try:
            getattr(axes, self.meth)(*self.args, zorder=z_order, **self.kwargs)
        except TypeError as e:
            if "zorder" in str(e):
                try:
                    getattr(axes, self.meth)(*self.args, **self.kwargs)
                except Exception as e2:
                    raise GraphingException(
                        f"Failed to call Axes method '{self.meth}' with provided arguments. Please check that all "
                        "provided arguments are valid for the given method."
                    ) from e2
            else:
                raise GraphingException(
                    f"Failed to call Axes method '{self.meth}' with provided arguments. Please check that all "
                    "provided arguments are valid for the given method."
                ) from e
