"""WorldSynth: synthetic training data for robotic machine tending from CAD files.

Stage 1 renders labelled clips of randomized CAD scenes in Isaac Sim (``scripts/generate.py``).
Stage 2 makes them photorealistic with Cosmos Transfer while keeping the labels (``scripts/cosmos_prepare.py`` and
``scripts/cosmos_import.py``).

Modules that need Isaac Sim (:mod:`worldsynth.scene`, :mod:`worldsynth.capture`, :mod:`worldsynth.generate`) are not
imported here so the rest of the package works in a plain Python environment.
"""

__version__ = "0.1.0"
