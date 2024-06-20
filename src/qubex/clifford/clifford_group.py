from __future__ import annotations

import json
import random
from pathlib import Path

FILE_NAME = "clifford_group.json"


class Pauli:
    """
    Represents a Pauli operator.

    Attributes
    ----------
    coefficient : complex
        The coefficient of the Pauli operator. Must be one of ±1, ±i.
    operator : str
        The type of the Pauli operator. Must be one of 'I', 'X', 'Y', 'Z'.
    """

    def __init__(
        self,
        coefficient: complex,
        operator: str,
    ):
        if coefficient not in {1, -1, 1j, -1j}:
            raise ValueError("Invalid coefficient. Must be one of ±1, ±i.")
        if operator not in {"I", "X", "Y", "Z"}:
            raise ValueError(
                "Invalid Pauli operator. Must be one of 'I', 'X', 'Y', 'Z'."
            )
        self.coefficient = coefficient
        self.operator = operator

    def to_string(self) -> str:
        sign = {1: "", -1: "-", 1j: "i", -1j: "-i"}[self.coefficient]
        return f"{sign}{self.operator}"

    def print(self):
        print(self.to_string())

    def __repr__(self) -> str:
        return f"Pauli({self.coefficient}, '{self.operator}')"

    def __hash__(self):
        return hash((self.coefficient, self.operator))

    def __eq__(self, other) -> bool:
        return self.coefficient == other.coefficient and self.operator == other.operator


class Clifford:
    """
    Represents a Clifford operator.

    Attributes
    ----------
    name : str
        The name of the Clifford operator.
    map : dict[str, Pauli]
        The Pauli transformation map of the Clifford operator.
    """

    def __init__(
        self,
        name: str,
        map: dict[str, Pauli],
    ):
        self.name = name
        self.map = map

    @classmethod
    def identity(cls) -> Clifford:
        """
        Create the identity Clifford transformation.

        Returns
        -------
        Clifford
            An identity Clifford with no operations and a map that does not change Pauli operators.
        """
        return Clifford(
            name="I",
            map={
                "I": Pauli(1, "I"),
                "X": Pauli(1, "X"),
                "Y": Pauli(1, "Y"),
                "Z": Pauli(1, "Z"),
            },
        )

    @classmethod
    def x90(cls) -> Clifford:
        """
        Create the Clifford transformation for a 90-degree rotation around the X-axis.

        Returns
        -------
        Clifford
            A Clifford representing a 90-degree rotation around the X-axis.
        """
        return Clifford(
            name="X90",
            map={
                "I": Pauli(1, "I"),
                "X": Pauli(1, "X"),
                "Y": Pauli(1, "Z"),
                "Z": Pauli(-1, "Y"),
            },
        )

    @classmethod
    def z90(cls) -> Clifford:
        """
        Create the Clifford transformation for a 90-degree rotation around the Z-axis.

        Returns
        -------
        Clifford
            A Clifford representing a 90-degree rotation around the Z-axis.
        """
        return Clifford(
            name="Z90",
            map={
                "I": Pauli(1, "I"),
                "X": Pauli(1, "Y"),
                "Y": Pauli(-1, "X"),
                "Z": Pauli(1, "Z"),
            },
        )

    @property
    def inverse(self) -> Clifford:
        """
        Compute the inverse of the Clifford transformation.

        Returns
        -------
        Clifford
            The inverse of the current Clifford transformation.
        """
        inverse_map = {}
        for operator, pauli in self.map.items():
            coefficient = 1 / pauli.coefficient
            inverse_map[pauli.operator] = Pauli(coefficient, operator)
        map = {
            "I": inverse_map["I"],
            "X": inverse_map["X"],
            "Y": inverse_map["Y"],
            "Z": inverse_map["Z"],
        }
        return Clifford(
            name=f"{self.name}^-1",
            map=map,
        )

    def is_identity(self) -> bool:
        """Check if the Clifford transformation is the identity."""
        return all(pauli.coefficient == 1 for pauli in self.map.values())

    def compose(
        self,
        other: Clifford,
    ) -> Clifford:
        """
        Compose two Clifford transformations.

        Parameters
        ----------
        other : Clifford
            The other Clifford transformation to compose with.

        Returns
        -------
        Clifford
            The resulting Clifford transformation after composing the two input transformations.
        """
        composed_map = {}
        for operator, pauli in self.map.items():
            composed_map[operator] = other.apply_to(pauli)
        return Clifford(
            name=f"{self.name}->{other.name}",
            map=composed_map,
        )

    def apply_to(
        self,
        pauli: Pauli,
    ) -> Pauli:
        """
        Apply the Clifford transformation to a given Pauli operator.

        Parameters
        ----------
        pauli : Pauli
            The Pauli operator to transform.

        Returns
        -------
        Pauli
            The resulting Pauli operator after the transformation.
        """
        mapped_pauli = self.map[pauli.operator]
        new_coefficient = pauli.coefficient * mapped_pauli.coefficient
        return Pauli(new_coefficient, mapped_pauli.operator)

    def to_string(self) -> str:
        map = ", ".join(
            f"{operator}->{pauli.to_string()}" for operator, pauli in self.map.items()
        )
        return f"{{{map}}}"

    def to_dict(self) -> dict:
        return {operator: pauli.to_string() for operator, pauli in self.map.items()}

    def print(self):
        print(self.to_string())

    def __repr__(self) -> str:
        return f"Clifford(name='{self.name}', map={self.to_string()})"

    def __hash__(self):
        return hash(tuple(self.map.items()))

    def __eq__(self, other) -> bool:
        return self.map == other.map


class CliffordSequence:
    """
    Represents a sequence of Clifford operators.

    Attributes
    ----------
    sequence : list[Clifford]
        The sequence of Clifford operators.
    clifford : Clifford
        The cumulative Clifford operator of the sequence.
    """

    def __init__(
        self,
        sequence: list[Clifford],
        clifford: Clifford,
    ):
        self.sequence = sequence
        self.clifford = clifford
        self._inverse: CliffordSequence | None = None

    @classmethod
    def identity(cls) -> CliffordSequence:
        """
        Create an identity Clifford sequence.

        Returns
        -------
        CliffordSequence
            An identity Clifford sequence with no operations and an identity Clifford.
        """
        return cls(sequence=[], clifford=Clifford.identity())

    @property
    def gate_sequence(self) -> list[str]:
        """Returns the sequence of gate names."""
        return [clifford.name for clifford in self.sequence]

    @property
    def inverse(self) -> CliffordSequence:
        """
        Compute the inverse of the Clifford sequence.

        Returns
        -------
        CliffordSequence
            The inverse of the current Clifford sequence.
        """
        if self._inverse is None:
            raise ValueError("No inverse set.")
        return self._inverse

    @inverse.setter
    def inverse(self, inverse: CliffordSequence):
        self._inverse = inverse

    @property
    def length(self) -> int:
        """Returns the length of the sequence."""
        return len(self.sequence)

    def count(self, clifford: Clifford) -> int:
        """
        Count the number of occurrences of a Clifford operator in the sequence.

        Parameters
        ----------
        clifford : Clifford
            The Clifford operator to count.

        Returns
        -------
        int
            The number of occurrences of the input Clifford operator in the sequence.
        """
        return self.sequence.count(clifford)

    def compose(
        self,
        other: Clifford,
    ) -> CliffordSequence:
        """
        Compose a Clifford transformation with the current sequence.

        Parameters
        ----------
        other : Clifford
            The Clifford transformation to compose with the sequence.

        Returns
        -------
        CliffordSequence
            The resulting Clifford sequence after composing the input transformation.
        """
        composed_sequence = self.sequence + [other]
        composed_clifford = self.clifford.compose(other)
        return CliffordSequence(sequence=composed_sequence, clifford=composed_clifford)

    def __hash__(self) -> int:
        return hash(tuple(self.sequence))


class CliffordGroup:
    def __init__(self):
        self._clifford_sequences = list[CliffordSequence]()

    @property
    def cliffords(self) -> dict[str, dict]:
        """Returns the Clifford operators in the group."""
        return self.load()

    def get_clifford(self, index: int) -> dict[str, dict]:
        """Returns the Clifford operator at a given index."""
        return self.cliffords[f"#{index}"]

    def get_random_cliffords(
        self,
        n: int,
        seed: int = 0,
    ) -> list[dict]:
        """Returns a list of n random Clifford operators."""
        random.seed(seed)
        keys = list(self.cliffords.keys())
        random_keys = random.choices(keys, k=n)
        return [self.cliffords[key] for key in random_keys]

    def generate(
        self,
        max_gates: int = 5,
    ):
        """
        Generate unique Clifford sequences up to a specified gate count.

        Parameters
        ----------
        max_gates : int
            The maximum number of gates in the Clifford sequences.

        Returns
        -------
        set[CliffordSequence]
            The set of unique Clifford sequences up to the specified gate count.
        """
        # Start with the identity Clifford
        identity = CliffordSequence.identity()

        # Dictionary to store found Clifford operators
        found_cliffords: dict[Clifford, CliffordSequence] = {
            identity.clifford: identity
        }

        # Clifford generators
        x90 = Clifford.x90()
        z90 = Clifford.z90()

        # Recursive function to generate Clifford sequences
        def generate_clifford_sequences(
            sequence: CliffordSequence,
            gate_count: int,
        ):
            if gate_count == 0:
                # Base case: reached the maximum number of gates
                return
            for clifford in (x90, z90):
                # Compose the current sequence with the Clifford generator
                new_sequence = sequence.compose(clifford)
                # Add the new sequence to the dictionary if it is unique
                if new_sequence.clifford not in found_cliffords:
                    found_cliffords[new_sequence.clifford] = new_sequence
                # Update the existing sequence if the new sequence is shorter
                else:
                    existing_sequence = found_cliffords[new_sequence.clifford]
                    if new_sequence.count(x90) < existing_sequence.count(x90):
                        found_cliffords[new_sequence.clifford] = new_sequence
                # Recurse with the new sequence
                generate_clifford_sequences(new_sequence, gate_count - 1)

        # Generate Clifford sequences starting from the identity
        generate_clifford_sequences(identity, max_gates + 1)  # +1 for the identity

        # Set the inverse of each Clifford sequence
        for sequence in found_cliffords.values():
            inverse = found_cliffords.get(sequence.clifford.inverse, None)
            if inverse is None:
                raise ValueError("Inverse not found.")
            sequence.inverse = inverse

        # Store the Clifford sequences in the group
        clifford_sequences = list(found_cliffords.values())
        self._clifford_sequences = clifford_sequences

    def save(self):
        """Save the Clifford group to a JSON file."""
        data = dict[str, dict]()
        for index, clifford_sequence in enumerate(self._clifford_sequences):
            data[f"#{index}"] = {
                "index": index,
                "map": clifford_sequence.clifford.to_dict(),
                "sequence": clifford_sequence.gate_sequence,
                "inverse": clifford_sequence.inverse.gate_sequence,
            }

        current_dir = Path(__file__).parent
        file_path = current_dir / FILE_NAME
        with open(file_path, "w") as file:
            json.dump(data, file, indent=2)

    def load(self) -> dict[str, dict]:
        """Load the Clifford group from a JSON file."""
        current_dir = Path(__file__).parent
        file_path = current_dir / FILE_NAME
        with open(file_path, "r") as file:
            return json.load(file)
