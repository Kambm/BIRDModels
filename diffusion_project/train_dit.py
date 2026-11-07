"""DiT training entry point: python -m diffusion_project.train_dit --config FILE."""

from .training import main

if __name__ == "__main__":
    main("dit")
