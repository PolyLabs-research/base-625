# Spatial Brain Overview

This playground explores spatial computing ideas: separating content from control and using spatial fields as the control interface.

- Content lives in cortical columns as continuous vectors (rate-coded spiking).
- Control uses beta stencils to open/close gates over space; gamma amplifies expression and is anti-correlated with beta.
- Working-memory slots are patches on the cortical sheet; opening a patch lets content land and be maintained.

Current implementation:

- A 1D cortical line with simple recurrent and lateral dynamics.
- Control network generates beta (and derived gamma) given context and time.
- A three-slot ordered working-memory task drives beta stencils to carve slot-specific regions.
- A 2D cortical sheet variant mirrors the same API to prepare for glyph-based tasks.

Future directions:

- Expand to a 2D cortical sheet to host 25x25 base-625 glyphs.
- Integrate base-625 boolean glyph grids as spatial stimuli and link to more symbolic tasks such as first-order logic operations.
- Richer temporal dynamics and visualizations of beta/gamma propagation.
