"""
Utility modules for in-context few-shot learning in chemical space.
"""

from .similarity import (
    SIMILARITY_FUNCTIONS,
    get_similarity_function,
    compute_similarity_matrix,
    compute_multiple_similarities,
    morgan_similarity,
    maccs_similarity,
    rdkit_similarity,
)

from .shape_color_descriptors import (
    generate_3d_conformer,
    compute_rmsd,
    rmsd_similarity,
    pharmacophore_similarity,
    espsim_similarity,
    compute_shape_color_combined,
)

from .shape_color_similarity import (
    SHAPE_COLOR_METHODS,
    compute_shape_color_similarities,
    prepare_molecules_3d,
)

from .metagraph import (
    build_metagraph,
    compute_metagraph_statistics,
)

__all__ = [
    'SIMILARITY_FUNCTIONS',
    'SHAPE_COLOR_METHODS',
    'get_similarity_function',
    'compute_similarity_matrix',
    'compute_multiple_similarities',
    'morgan_similarity',
    'maccs_similarity',
    'rdkit_similarity',
    'generate_3d_conformer',
    'compute_rmsd',
    'rmsd_similarity',
    'pharmacophore_similarity',
    'espsim_similarity',
    'compute_shape_color_combined',
    'compute_shape_color_similarities',
    'prepare_molecules_3d',
    'build_metagraph',
    'compute_metagraph_statistics',
]
