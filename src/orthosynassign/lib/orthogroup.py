from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from .gene import Gene


class Orthogroup:
    """Represents a group of orthologous genes.

    Attributes:
        id: The unique identifier for the orthogroup.
        _genes: A private attribute to store the genes in the orthogroup.
    """

    __slots__ = ("id", "_genes")

    def __init__(self, og_id: str | None = None, genes: list[Gene] | None = None) -> None:
        """Initialize an Orthogroup.

        Args:
            og_id (str): The unique identifier for the orthogroup.
            genes (list[Gene]): The genes in the orthogroup.
        """
        self.id = og_id
        self._genes: list[Gene] = []
        if genes:
            for gene in genes:
                self.add_gene(gene)

    def __repr__(self) -> str:
        """Return a string representation of the Orthogroup.

        Returns:
            str: A string in the format "[{id} | with {len(self._genes)} genes]".
        """
        og_id = self.id if self.id else "Unnamed orthogroup"
        return f"[{og_id} | with {len(self._genes)} genes]"

    def __len__(self) -> int:
        """Return the number of genes in the Orthogroup.

        Returns:
            int: The number of genes.
        """
        return len(self._genes)

    def __getitem__(self, index: int) -> Gene:
        """Retrieve a gene by its index in the Orthogroup.

        Args:
            index (int): The index of the gene to retrieve.

        Returns:
            Gene: The gene at the specified index.
        """
        return self._genes[index]

    def __contains__(self, item: Gene) -> bool:
        """Allows for 'gene in orthogroup' syntax."""
        return item in self._genes

    def __iter__(self) -> Iterator[Gene]:
        """Return an iterator over the genes in the Orthogroup.

        Returns:
            Iterator[Gene]: An iterator over the genes.
        """
        return iter(self._genes)

    def __getstate__(self) -> dict[str, Any]:
        """Get the state of the object for pickling.

        Returns:
            dict[str, Any]: The state of the object.
        """
        return {slot: getattr(self, slot) for slot in self.__slots__ if hasattr(self, slot)}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the state of the object from pickling.

        Args:
            state (dict[str, Any]): The state to restore.
        """
        for slot, value in state.items():
            setattr(self, slot, value)

    def add_gene(self, gene: Gene) -> None:
        """Add a gene to the Orthogroup.

        If the gene is not already present in the Orthogroup, it will be added and its og attribute set to this Orthogroup.

        Args:
            gene (Gene): The gene to add.
        """
        if gene not in self._genes:
            gene.og = self
            self._genes.append(gene)
