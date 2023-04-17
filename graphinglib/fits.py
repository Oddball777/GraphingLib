from tracemalloc import Statistic

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from .data_plotting_1d import Curve


class FitFromPolynomial(Curve):
    """
    Create a curve fit (continuous Curve) from an existing curve object using a polynomial fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve,
        degree: int,
        label: str,
        color: str = "default",
        line_width: int = "default",
    ):
        self.curve_to_be_fit = curve_to_be_fit
        inversed_coeffs, inversed_cov_matrix = np.polyfit(
            self.curve_to_be_fit.xdata, self.curve_to_be_fit.ydata, degree, cov=True
        )
        self.coeffs = inversed_coeffs[::-1]
        self.cov_matrix = np.flip(inversed_cov_matrix)
        self.standard_deviation = np.sqrt(np.diag(self.cov_matrix))
        self.function = self.polynomial_func_with_params()
        self.color = color
        self.line_width = line_width
        self.label = label + " : " + "f(x) = " + str(self)

    def __str__(self):
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
        return "".join(
            [coeff_chunks[i] + power_chunks[i] for i in range(len(coeff_chunks))]
        )

    @staticmethod
    def format_coeff(coeff):
        return " - {0}".format(abs(coeff)) if coeff < 0 else " + {0}".format(coeff)

    @staticmethod
    def format_power(power):
        return "x^{0}".format(power) if power != 0 else ""

    def polynomial_func_with_params(self):
        """
        Returns a linear function using the class' coefficients.
        """
        return lambda x: sum(
            coeff * x**exponent for exponent, coeff in enumerate(self.coeffs)
        )

    def plot_element(self, axes: plt.Axes):
        num_of_points = 500
        xdata = np.linspace(
            self.curve_to_be_fit.xdata[0], self.curve_to_be_fit.xdata[-1], num_of_points
        )
        ydata = self.function(xdata)
        (self.handle,) = axes.plot(
            xdata, ydata, label=self.label, color=self.color, linewidth=self.line_width
        )


class FitFromSine(Curve):
    """
    Create a curve fit (continuous Curve) from an existing curve object using a sinusoidal fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve,
        color: str,
        label: str,
        guesses: npt.ArrayLike = None,
    ):
        self.curve_to_be_fit = curve_to_be_fit
        self.guesses = guesses
        self.calculate_parameters()
        self.function = self.sine_func_with_params()
        self.color = color
        self.label = label + " : " + "f(x) = " + str(self)

    def __str__(self) -> str:
        part1 = f"{self.amplitude:.3f} sin({self.frequency_rad:.3f}x"
        part2 = (
            f" + {self.phase:.3f})" if self.phase >= 0 else f" - {abs(self.phase):.3f})"
        )
        part3 = (
            f" + {self.vertical_shift:.3f}"
            if self.vertical_shift >= 0
            else f" - {abs(self.vertical_shift):.3f}"
        )
        return part1 + part2 + part3

    def calculate_parameters(self):
        parameters, self.cov_matrix = curve_fit(
            self.sine_func_template,
            self.curve_to_be_fit.xdata,
            self.curve_to_be_fit.ydata,
            p0=self.guesses,
        )
        self.amplitude, self.frequency_rad, self.phase, self.vertical_shift = parameters
        self.standard_deviation = np.sqrt(np.diag(self.cov_matrix))

    @staticmethod
    def sine_func_template(x, a, b, c, d):
        return a * np.sin(b * x + c) + d

    def sine_func_with_params(self):
        return (
            lambda x: self.amplitude * np.sin(self.frequency_rad * x + self.phase)
            + self.vertical_shift
        )

    def plot_element(self, axes: plt.Axes):
        num_of_points = 500
        xdata = np.linspace(
            self.curve_to_be_fit.xdata[0], self.curve_to_be_fit.xdata[-1], num_of_points
        )
        ydata = self.function(xdata)
        (self.handle,) = axes.plot(xdata, ydata, color=self.color, label=self.label)


class FitFromExponential(Curve):
    """
    Create a curve fit (continuous Curve) from an existing curve object using a sinusoidal fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve,
        color: str,
        label: str,
        guesses: npt.ArrayLike = None,
    ):
        self.curve_to_be_fit = curve_to_be_fit
        self.guesses = guesses
        self.calculate_parameters()
        self.function = self.exp_func_with_params()
        self.color = color
        self.label = label + " : " + "f(x) = " + str(self)

    def __str__(self) -> str:
        part1 = f"{self.parameters[0]:.3f} exp({self.parameters[1]:.3f}x"
        part2 = (
            f" + {self.parameters[2]:.3f})"
            if self.parameters[2] >= 0
            else f" - {abs(self.parameters[2]):.3f})"
        )
        return part1 + part2

    def calculate_parameters(self):
        parameters, self.cov_matrix = curve_fit(
            self.exp_func_template,
            self.curve_to_be_fit.xdata,
            self.curve_to_be_fit.ydata,
            p0=self.guesses,
        )
        self.parameters = parameters
        self.standard_deviation = np.sqrt(np.diag(self.cov_matrix))

    @staticmethod
    def exp_func_template(x, a, b, c):
        return a * np.exp(b * x + c)

    def exp_func_with_params(self):
        return lambda x: self.parameters[0] * np.exp(
            self.parameters[1] * x + self.parameters[2]
        )

    def plot_element(self, axes: plt.Axes):
        num_of_points = 500
        xdata = np.linspace(
            self.curve_to_be_fit.xdata[0], self.curve_to_be_fit.xdata[-1], num_of_points
        )
        ydata = self.function(xdata)
        (self.handle,) = axes.plot(xdata, ydata, color=self.color, label=self.label)


class FitFromGaussian(Curve):
    """
    Create a curve fit (continuous Curve) from an existing curve object using a sinusoidal fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve,
        color: str,
        label: str,
        guesses: npt.ArrayLike = None,
    ):
        self.curve_to_be_fit = curve_to_be_fit
        self.guesses = guesses
        self.calculate_parameters()
        self.function = self.gaussian_func_with_params()
        self.color = color
        self.label = label + " : " + "f(x) = " + str(self)

    def __str__(self) -> str:
        return "mock_string"

    def calculate_parameters(self):
        parameters, self.cov_matrix = curve_fit(
            self.gaussian_func_template,
            self.curve_to_be_fit.xdata,
            self.curve_to_be_fit.ydata,
            p0=self.guesses,
        )
        self.amplitude = parameters[0]
        self.mean = parameters[1]
        self.standard_deviation = parameters[2]
        self.standard_deviation_of_fit_params = np.sqrt(np.diag(self.cov_matrix))

    @staticmethod
    def gaussian_func_template(x, amplitude, mean, standard_deviation):
        return amplitude * np.exp(-(((x - mean) / 4 / standard_deviation) ** 2))

    def gaussian_func_with_params(self):
        return lambda x: self.amplitude * np.exp(
            -(((x - self.mean) / 4 / self.standard_deviation) ** 2)
        )

    def plot_element(self, axes: plt.Axes):
        num_of_points = 500
        xdata = np.linspace(
            self.curve_to_be_fit.xdata[0], self.curve_to_be_fit.xdata[-1], num_of_points
        )
        ydata = self.function(xdata)
        (self.handle,) = axes.plot(xdata, ydata, color=self.color, label=self.label)
