"""
Dieses Modul enthält eine einfache, aber vollständige Implementierung
eines mehrschichtigen Feedforward-Neuronalen Netzes in reinem NumPy.

Das Netzwerk besteht aus:
  * Beliebig vielen Schichten, die jeweils eine Gewichtsmatrix und einen Bias-Vektor besitzen.
  * Aktivierungsfunktionen (Sigmoid für die versteckten Schichten und Softmax für die Ausgangsschicht).
  * Einer vollständigen Vorwärts- und Rückwärts-Propagation einschließlich Cross-Entropy-Loss.

Das Beispiel zeigt, wie man das Netzwerk mit zufällig generierten Daten
trainieren und anschließend für Vorhersagen verwenden kann.

Hinweis: Diese Implementierung verwendet keine externen Bibliotheken
wie sklearn oder transformers und ist damit portabel in jeder
Standard-Python-Umgebung mit NumPy.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Tuple


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid-Aktivierungsfunktion für versteckte Schichten."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Ableitung der Sigmoidfunktion nach ihrer Eingabe."""
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax-Funktion, die einen Vektor in Wahrscheinlichkeiten überführt."""
    # Für numerische Stabilität wird der Maximalwert subtrahiert
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Berechnet den Cross-Entropy-Loss zwischen Vorhersage und Ziel (One-Hot)."""
    # Clip verhindert log(0)
    y_pred_clipped = np.clip(y_pred, 1e-12, 1.0)
    return -float(np.sum(y_true * np.log(y_pred_clipped)))


@dataclass
class Layer:
    """
    Repräsentiert eine einzelne Schicht eines Feedforward-Netzwerks.

    Die Schicht enthält eine Gewichtsmatrix (shape: [input_dim, output_dim]),
    einen Bias-Vektor (shape: [output_dim]) und Aktivierungsfunktionen.
    """

    input_dim: int
    output_dim: int
    activation: Callable[[np.ndarray], np.ndarray]
    activation_derivative: Callable[[np.ndarray], np.ndarray]
    weights: np.ndarray = field(init=False)
    biases: np.ndarray = field(init=False)
    z: np.ndarray = field(init=False, default=None)
    a: np.ndarray = field(init=False, default=None)

    def __post_init__(self) -> None:
        # Initialisieren der Gewichte mit kleiner Zufallsverteilung (He- oder Xavier-Init)
        limit = np.sqrt(2 / self.input_dim)
        self.weights = np.random.randn(self.input_dim, self.output_dim) * limit
        self.biases = np.zeros(self.output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Berechnet z = Wx + b und die Aktivierung a."""
        self.z = np.dot(x, self.weights) + self.biases
        self.a = self.activation(self.z)
        return self.a

    def backward(self, grad_w: np.ndarray, grad_b: np.ndarray, learning_rate: float) -> None:
        """
        Aktualisiert die Gewichte und Biases dieser Schicht.

        Der Fehlerterm (delta) wird außerhalb der Schicht berechnet, um die
        Kettenregel korrekt anwenden zu können. Diese Methode nimmt lediglich
        die Gradienten für Gewichte und Biases entgegen und führt das Update
        durch.
        """
        self.weights -= learning_rate * grad_w
        self.biases -= learning_rate * grad_b


@dataclass
class NeuralNetwork:

    input_dim: int
    hidden_layers_dims: List[int]
    output_dim: int
    learning_rate: float = 0.1
    layers: List[Layer] = field(init=False)

    def __post_init__(self) -> None:
        self.layers = []
        prev_dim = self.input_dim
        # Versteckte Schichten mit Sigmoid-Aktivierung
        for dim in self.hidden_layers_dims:
            self.layers.append(
                Layer(
                    input_dim=prev_dim,
                    output_dim=dim,
                    activation=sigmoid,
                    activation_derivative=sigmoid_derivative,
                )
            )
            prev_dim = dim
        # Ausgangsschicht mit Softmax; derivative wird separat behandelt
        self.layers.append(
            Layer(
                input_dim=prev_dim,
                output_dim=self.output_dim,
                activation=lambda x: x,  # lineare Aktivierung; Softmax separat
                activation_derivative=lambda x: np.ones_like(x),
            )
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = x
        for i, layer in enumerate(self.layers):
            z = np.dot(a, layer.weights) + layer.biases
            layer.z = z
            # Für die letzte Schicht wenden wir Softmax erst am Ende an
            if i == len(self.layers) - 1:
                layer.a = z  # lineare Ausgabe speichern
                a = z
            else:
                a = layer.activation(z)
                layer.a = a
        return softmax(a)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray, x: np.ndarray) -> None:
        """
        Führt die Backpropagation aus und aktualisiert die Gewichte.

        y_pred: Vorhersage (Softmax-Ausgabe)
        y_true: Ziel als One-Hot-Vektor
        x: Originale Eingabe
        """
        # Start-Delta für die Ausgangsschicht (Softmax + Cross-Entropy): y_pred - y_true
        delta = y_pred - y_true
        # Rückwärts durch die Schichten iterieren
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            prev_a = x if i == 0 else self.layers[i - 1].a
            # Für die Berechnung des delta der vorherigen Schicht benötigen wir die aktuellen Gewichte
            # Wir speichern eine Kopie, damit die Aktualisierung den Gradientenfluss nicht beeinflusst
            weights_copy = layer.weights.copy()
            # Gradienten berechnen
            grad_w = np.outer(prev_a, delta)
            grad_b = delta
            # Gewichte und Biases aktualisieren
            layer.backward(grad_w, grad_b, self.learning_rate)
            # Delta für die vorherige Schicht berechnen, außer wir sind bei der Eingabeschicht
            if i > 0:
                delta = np.dot(weights_copy, delta) * self.layers[i - 1].activation_derivative(self.layers[i - 1].z)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
    ) -> List[float]:
        """
        Trainiert das Netz per Stochastic Gradient Descent.
        Gibt die Entwicklung des Verlustes pro Epoche zurück.
        """
        losses: List[float] = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                # Vorwärts-Pass
                y_pred = self.forward(x)
                # Verlust berechnen
                epoch_loss += cross_entropy_loss(y_pred, y_true)
                # Rückwärts-Pass
                self.backward(y_pred, y_true, x)
            losses.append(epoch_loss / len(X))
        return losses

    def predict(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Liefert die vorhergesagte Klasse und die Softmax-Wahrscheinlichkeiten für ein Eingabevektor.
        """
        probs = self.forward(x)
        return int(np.argmax(probs)), probs


def _demo() -> None:
    """Demonstrationsfunktion für das Training auf einem synthetischen Datensatz."""
    # Erzeuge zufälligen Datensatz für binäre Klassifikation mit zwei Merkmalen
    np.random.seed(42)
    num_samples = 200
    X = np.random.randn(num_samples, 2)

    # Ziel: Klasse 1, wenn x1 + x2 > 0, sonst Klasse 0
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # shuffle vor Split
    perm = np.random.permutation(num_samples)
    X, y = X[perm], y[perm]

    # 10% Label-Rauschen
    noise_rate = 0.1
    flip = np.random.rand(num_samples) < noise_rate
    y_noisy = y.copy()
    y_noisy[flip] = 1 - y_noisy[flip]

    # One-Hot-Kodierung **aus y_noisy**
    y_one_hot = np.zeros((num_samples, 2))
    y_one_hot[np.arange(num_samples), y_noisy] = 1

    # Train/Test-Split
    split = int(0.8 * num_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_one_hot[:split], y_one_hot[split:]

    # Netz (du kannst epochs/learning_rate gern kleiner wählen, wenn’s zu 1.00 wird)
    net = NeuralNetwork(input_dim=2, hidden_layers_dims=[3], output_dim=2, learning_rate=0.1)

    # Training
    losses = net.train(X_train, y_train, epochs=500)

    # Grobe Evaluation
    correct = 0
    for i in range(len(X_test)):
        pred, _ = net.predict(X_test[i])
        if pred == int(np.argmax(y_test[i])):
            correct += 1
    print("Test-Accuracy:", correct / len(X_test))


if __name__ == "__main__":
    _demo()