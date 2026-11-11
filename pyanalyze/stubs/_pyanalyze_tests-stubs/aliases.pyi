from typing_extensions import TypeAlias

Alias = int
ExplicitAlias: TypeAlias = int
type Pep695Alias = int

constant: int
aliased_constant: Alias
explicitly_aliased_constant: ExplicitAlias
pep695_aliased_constant: Pep695Alias
