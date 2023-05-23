from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from .data_plotting_1d import Curve, Scatter


class GeneralFit(Curve):
    def __init__(
        self,
        curve_to_be_fit: Curve,
        label: str = None,
        color: str = "default",
        line_width: int = "default",
        line_style: int = "default",
    ) -> None:
        self.curve_to_be_fit = curve_to_be_fit
        self.color = color
        self.line_width = line_width
        if label:
            self.label = label + " : " + "$f(x) = $" + str(self)
        else:
            self.label = "$f(x) = $" + str(self)
        self.line_style = line_style
        self._res_curves_to_be_plotted = False

    def __str__(self) -> None:
        raise NotImplementedError()

    def plot_element(self, axes: plt.Axes, z_order: int) -> None:
        num_of_points = 500
        xdata = np.linspace(
            self.curve_to_be_fit.x_data[0],
            self.curve_to_be_fit.x_data[-1],
            num_of_points,
        )
        ydata = self.function(xdata)
        (self.handle,) = axes.plot(
            xdata,
            ydata,
            label=self.label,
            color=self.color,
            linewidth=self.line_width,
            linestyle=self.line_style,
            zorder=z_order,
        )
        if self._res_curves_to_be_plotted:
            xdata = self.curve_to_be_fit.x_data
            yfit = self.function(xdata)
            residuals = self.calculate_residuals()
            std = np.std(residuals)
            y_fit_plus_std = yfit + (self.res_sigma_multiplier * std)
            y_fit_minus_std = yfit - (self.res_sigma_multiplier * std)
            axes.plot(
                xdata,
                y_fit_minus_std,
                label=self.label,
                color=self.res_color,
                linewidth=self.res_line_width,
                linestyle=self.res_line_style,
                zorder=z_order,
            )
            axes.plot(
                xdata,
                y_fit_plus_std,
                label=self.label,
                color=self.res_color,
                linewidth=self.res_line_width,
                linestyle=self.res_line_style,
                zorder=z_order,
            )

    def show_residual_curves(
        self,
        sigma_multiplier: float = 1,
        color: str = "default",
        line_width: float = "default",
        line_style: str = "default",
    ) -> None:
        self._res_curves_to_be_plotted = True
        self.res_sigma_multiplier = sigma_multiplier
        self.res_color = color
        self.res_line_width = line_width
        self.res_line_style = line_style

    def calculate_residuals(self) -> np.ndarray:
        xdata = self.curve_to_be_fit.x_data
        ydata = self.curve_to_be_fit.y_data
        yfit = self.function(xdata)
        residuals = yfit - ydata
        return residuals


class FitFromPolynomial(GeneralFit):
    """
    Create a curve fit (continuous Curve) from an existing curve object using a polynomial fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve,
        degree: int,
        label: str = None,
        color: str = "default",
        line_width: int = "default",
        line_style: int = "default",
    ) -> None:
        self.curve_to_be_fit = curve_to_be_fit
        inversed_coeffs, inversed_cov_matrix = np.polyfit(
            self.curve_to_be_fit.x_data, self.curve_to_be_fit.y_data, degree, cov=True
        )
        self.coeffs = inversed_coeffs[::-1]
        self.cov_matrix = np.flip(inversed_cov_matrix)
        self.standard_deviation = np.sqrt(np.diag(self.cov_matrix))
        self.function = self.polynomial_func_with_params()
        self.color = color
        self.line_width = line_width
        if label:
            self.label = label + " : " + "$f(x) = $" + str(self)
        else:
            self.label = "$f(x) = $" + str(self)
        self.line_style = line_style
        self._res_curves_to_be_plotted = False

    def __str__(self) -> str:
        coeff_chunks = []
        power_chunks = []
        ordered_rounded_coeffs = [round(coeff, 3) for coeff in self.coeffs[::-1]]
        for coeff, power in zip(
            ordered_rounded_coeffs, range(len(ordered_rounded_coeffs) - 1, -1, -1)
        ):
            if coeff == 0:
                continue
            coeff_chunks.append(self.format_coeff(coeff))
            power_chunks.append(self.format_power(power))
        coeff_chunks[0] = coeff_chunks[0].lstrip("+ ")
        return (
            "$"
            + "".join(
                [coeff_chunks[i] + power_chunks[i] for i in range(len(coeff_chunks))]
            )
            + "$"
        )

    @staticmethod
    def format_coeff(coeff: float) -> str:
        return " - {0}".format(abs(coeff)) if coeff < 0 else " + {0}".format(coeff)

    @staticmethod
    def format_power(power: int) -> str:
        return "x^{0}".format(power) if power != 0 else ""

    def polynomial_func_with_params(self) -> Callable:
        """
        Returns a linear function using the class' coefficients.
        """
        return lambda x: sum(
            coeff * x**exponent for exponent, coeff in enumerate(self.coeffs)
        )


class FitFromSine(GeneralFit):
    """
    Create a curve fit (continuous Curve) from an existing curve object using a sinusoidal fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve,
        label: str = None,
        guesses: npt.ArrayLike = None,
        color: str = "default",
        line_width: str = "default",
        line_style: str = "default",
    ) -> None:
        self.curve_to_be_fit = curve_to_be_fit
        self.guesses = guesses
        self.calculate_parameters()
        self.function = self.sine_func_with_params()
        self.color = color
        if label:
            self.label = label + " : " + "$f(x) = $" + str(self)
        else:
            self.label = "$f(x) = $" + str(self)
        self.line_width = line_width
        self.line_style = line_style
        self._res_curves_to_be_plotted = False

    def __str__(self) -> str:
        part1 = f"{self.amplitude:.3f} \sin({self.frequency_rad:.3f}x"
        part2 = (
            f" + {self.phase:.3f})" if self.phase >= 0 else f" - {abs(self.phase):.3f})"
        )
        part3 = (
            f" + {self.vertical_shift:.3f}"
            if self.vertical_shift >= 0
            else f" - {abs(self.vertical_shift):.3f}"
        )
        return f"${part1 + part2 + part3}$"

    def calculate_parameters(self) -> None:
        parameters, self.cov_matrix = curve_fit(
            self.sine_func_template,
            self.curve_to_be_fit.x_data,
            self.curve_to_be_fit.y_data,
            p0=self.guesses,
        )
        self.amplitude, self.frequency_rad, self.phase, self.vertical_shift = parameters
        self.standard_deviation = np.sqrt(np.diag(self.cov_matrix))

    @staticmethod
    def sine_func_template(
        x: np.ndarray, a: float, b: float, c: float, d: float
    ) -> np.ndarray:
        return a * np.sin(b * x + c) + d

    def sine_func_with_params(self) -> Callable:
        return (
            lambda x: self.amplitude * np.sin(self.frequency_rad * x + self.phase)
            + self.vertical_shift
        )


class FitFromExponential(GeneralFit):
    """
    Create a curve fit (continuous Curve) from an existing curve object using an exponential fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve,
        label: str = None,
        guesses: npt.ArrayLike = None,
        color: str = "default",
        line_width: int = "default",
        line_style: str = "default",
    ) -> None:
        self.curve_to_be_fit = curve_to_be_fit
        self.guesses = guesses
        self.calculate_parameters()
        self.function = self.exp_func_with_params()
        self.color = color
        if label:
            self.label = label + " : " + "$f(x) = $" + str(self)
        else:
            self.label = "$f(x) = $" + str(self)
        self.line_width = line_width
        self.line_style = line_style
        self._res_curves_to_be_plotted = False

    def __str__(self) -> str:
        part1 = f"{self.parameters[0]:.3f} \exp({self.parameters[1]:.3f}x"
        part2 = (
            f" + {self.parameters[2]:.3f})"
            if self.parameters[2] >= 0
            else f" - {abs(self.parameters[2]):.3f})"
        )
        return f"${part1 + part2}$"

    def calculate_parameters(self) -> None:
        parameters, self.cov_matrix = curve_fit(
            self.exp_func_template,
            self.curve_to_be_fit.x_data,
            self.curve_to_be_fit.y_data,
            p0=self.guesses,
        )
        self.parameters = parameters
        self.standard_deviation = np.sqrt(np.diag(self.cov_matrix))

    @staticmethod
    def exp_func_template(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.exp(b * x + c)

    def exp_func_with_params(self) -> Callable:
        return lambda x: self.parameters[0] * np.exp(
            self.parameters[1] * x + self.parameters[2]
        )


class FitFromGaussian(GeneralFit):
    """
    Create a curve fit (continuous Curve) from an existing curve object using a gaussian fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve,
        label: str = None,
        guesses: npt.ArrayLike = None,
        color: str = "default",
        line_width: int = "default",
        line_style: str = "default",
    ) -> None:
        self.curve_to_be_fit = curve_to_be_fit
        self.guesses = guesses
        self.calculate_parameters()
        self.function = self.gaussian_func_with_params()
        self.color = color
        if label:
            self.label = label + " : " + str(self)
        else:
            self.label = str(self)
        self.line_width = line_width
        self.line_style = line_style
        self._res_curves_to_be_plotted = False

    def __str__(self) -> str:
        return f"$\mu = {self.mean:.3f}, \sigma = {self.standard_deviation:.3f}, A = {self.amplitude:.3f}$"

    def calculate_parameters(self) -> None:
        parameters, self.cov_matrix = curve_fit(
            self.gaussian_func_template,
            self.curve_to_be_fit.x_data,
            self.curve_to_be_fit.y_data,
            p0=self.guesses,
        )
        self.amplitude = parameters[0]
        self.mean = parameters[1]
        self.standard_deviation = parameters[2]
        self.standard_deviation_of_fit_params = np.sqrt(np.diag(self.cov_matrix))

    @staticmethod
    def gaussian_func_template(
        x: np.ndarray, amplitude: float, mean: float, standard_deviation: float
    ) -> np.ndarray:
        return amplitude * np.exp(-(((x - mean) / standard_deviation) ** 2) / 2)

    def gaussian_func_with_params(self) -> Callable:
        return lambda x: self.amplitude * np.exp(
            -(((x - self.mean) / self.standard_deviation) ** 2) / 2
        )


class FitFromSquareRoot(GeneralFit):
    """
    Create a curve fit (continuous Curve) from an existing curve object using a square root fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve,
        label: str = None,
        guesses: npt.ArrayLike = None,
        color: str = "default",
        line_width: int = "default",
        line_style: str = "default",
    ) -> None:
        self.curve_to_be_fit = curve_to_be_fit
        self.guesses = guesses
        self.calculate_parameters()
        self.function = self.square_root_func_with_params()
        self.color = color
        if label:
            self.label = label + " : " + str(self)
        else:
            self.label = str(self)
        self.line_width = line_width
        self.line_style = line_style
        self._res_curves_to_be_plotted = False

    def __str__(self) -> str:
        return f"${self.parameters[0]:.3f} \sqrt{{x {'+' if self.parameters[1] > 0 else '-'} {abs(self.parameters[1]):.3f}}} {'+' if self.parameters[2] > 0 else '-'} {abs(self.parameters[2]):.3f}$"

    def calculate_parameters(self) -> None:
        parameters, self.cov_matrix = curve_fit(
            self.square_root_func_template,
            self.curve_to_be_fit.x_data,
            self.curve_to_be_fit.y_data,
            p0=self.guesses,
        )
        self.parameters = parameters
        self.standard_deviation = np.sqrt(np.diag(self.cov_matrix))

    @staticmethod
    def square_root_func_template(
        x: np.ndarray, a: float, b: float, c: float
    ) -> np.ndarray:
        return a * np.sqrt(x + b) + c

    def square_root_func_with_params(self) -> Callable:
        return (
            lambda x: self.parameters[0] * np.sqrt(x + self.parameters[1])
            + self.parameters[2]
        )


class FitFromLog(GeneralFit):
    """
    Create a curve fit (continuous Curve) from an existing curve object using a logarithmic fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve,
        label: str = None,
        log_base: float = np.e,
        guesses: npt.ArrayLike = None,
        color: str = "default",
        line_width: int = "default",
        line_style: str = "default",
    ) -> None:
        self.curve_to_be_fit = curve_to_be_fit
        self.log_base = log_base
        self.guesses = guesses
        self.calculate_parameters()
        self.function = self.log_func_with_params()
        self.color = color
        if label:
            self.label = label + " : " + str(self)
        else:
            self.label = str(self)
        self.line_width = line_width
        self.line_style = line_style
        self._res_curves_to_be_plotted = False

    def __str__(self) -> str:
        return f"${self.parameters[0]:.3f} log_{self.log_base if self.log_base != np.e else 'e'}(x {'-' if self.parameters[1] < 0 else '+'} {abs(self.parameters[1]):.3f}) {'-' if self.parameters[2] < 0 else '+'} {abs(self.parameters[2]):.3f}$"

    def calculate_parameters(self) -> None:
        self.parameters, self.cov_matrix = curve_fit(
            self.log_func_template(),
            self.curve_to_be_fit.x_data,
            self.curve_to_be_fit.y_data,
            p0=self.guesses,
        )
        self.standard_deviation = np.sqrt(np.diag(self.cov_matrix))

    def log_func_template(self) -> Callable:
        return lambda x, a, b, c: a * (np.log(x + b) / np.log(self.log_base)) + c

    def log_func_with_params(self) -> Callable:
        return (
            lambda x: self.parameters[0]
            * (np.log(x + self.parameters[1]) / np.log(self.log_base))
            + self.parameters[2]
        )


class FitFromFunction(GeneralFit):
    """
    Create a curve fit (continuous Curve) from a curve object using an arbitrary function passed as an argument.
    """

    def __init__(
        self,
        function: Callable,
        curve_to_fit: Curve | Scatter,
        label: str = None,
        guesses: list[str] = None,
        color: str = "default",
        line_width: int = "default",
        line_style: str = "default",
    ):
        self.function_template = function
        self.curve_to_be_fit = curve_to_fit
        self.guesses = guesses
        self.color = color
        self.line_width = line_width
        self.line_style = line_style

        self.calculate_parameters()
        self.function = self.get_function_with_params()
        self.label = label
        self._res_curves_to_be_plotted = False

    def calculate_parameters(self) -> None:
        self.parameters, self.cov_matrix = curve_fit(
            self.function_template,
            self.curve_to_be_fit.x_data,
            self.curve_to_be_fit.y_data,
            p0=self.guesses,
        )
        self.standard_deviation = np.sqrt(np.diag(self.cov_matrix))

    def get_function_with_params(self) -> Callable:
        argument_names = self.function_template.__code__.co_varnames[
            : self.function_template.__code__.co_argcount
        ][1:]
        args_dict = {
            argument_names[i]: self.parameters[i] for i in range(len(argument_names))
        }
        return partial(self.function_template, **args_dict)
