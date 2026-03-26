"""Training recipe registry for in-weights experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingRecipe:
    """Describes a complete training recipe."""

    recipe_name: str
    description: str
    use_mixed_edge_and_path_batches: bool
    predict_full_path: bool


TRAINING_RECIPES = {
    "staged_full_path": TrainingRecipe(
        recipe_name="staged_full_path",
        description=("Stage 1 edge memorization, then stage 2 full-path finetuning."),
        use_mixed_edge_and_path_batches=False,
        predict_full_path=True,
    ),
    "mixed_full_path": TrainingRecipe(
        recipe_name="mixed_full_path",
        description=("Single-stage mixed edge/path training with full-path prediction."),
        use_mixed_edge_and_path_batches=True,
        predict_full_path=True,
    ),
    "staged_hardest_token": TrainingRecipe(
        recipe_name="staged_hardest_token",
        description=("Stage 1 edge memorization, then stage 2 hardest-token finetuning."),
        use_mixed_edge_and_path_batches=False,
        predict_full_path=False,
    ),
    "mixed_hardest_token": TrainingRecipe(
        recipe_name="mixed_hardest_token",
        description=("Single-stage mixed edge/path training with hardest-token prediction."),
        use_mixed_edge_and_path_batches=True,
        predict_full_path=False,
    ),
}


def get_training_recipe(recipe_name: str) -> TrainingRecipe:
    """Returns the recipe metadata for a recipe name.

    Args:
        recipe_name: Input parameter.

    Returns:
        object: Function return value.
    """
    if recipe_name not in TRAINING_RECIPES:
        valid = ", ".join(sorted(TRAINING_RECIPES))
        raise ValueError(f"Unknown training_recipe '{recipe_name}'. Valid: {valid}")
    return TRAINING_RECIPES[recipe_name]


def get_training_recipe_help_text() -> str:
    """Returns a compact text description for argparse help output.

    Args:
        None: This callable does not take external parameters.

    Returns:
        object: Function return value.
    """
    parts = []
    for recipe_name in sorted(TRAINING_RECIPES):
        recipe = TRAINING_RECIPES[recipe_name]
        parts.append(f"{recipe_name}: {recipe.description}")
    return " | ".join(parts)
