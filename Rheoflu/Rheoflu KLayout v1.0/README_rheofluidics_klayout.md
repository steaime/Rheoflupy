# Rheofluidics KLayout Tools

Channel shape generator based on:
> Milani et al., *Rheofluidics: frequency-dependent rheology of single drops*, arXiv:2601.07461 (2026)

---

## Files

| File | Purpose |
|------|---------|
| `rheofluidics_macro.lym` | Standalone macro with GUI dialog — generate a channel on demand |
| `rheofluidics_pcell_lib.lym` | PCell library — parametric cells usable from the Libraries panel |

---

## Parameters

All parameters use physical units (µm) at the interface. Internally converted to metres for the ODE solver, then to DBU for KLayout.

| Parameter | Symbol in paper | Description |
|-----------|----------------|-------------|
| `L0` | L₀ | Constriction half-width at channel minimum [µm] |
| `sigma_tilde` | σ̃ = σ₀L₀²/(qη) | Dimensionless stress amplitude |
| `omega_tilde` | ω̃ = ωL₀²/q | Dimensionless frequency |
| `n_periods` | — | Number of oscillation periods per segment |
| `channel_length` | — | Total channel length [µm] (sets the flow rate q implicitly) |
| `npts` | — | Points per segment (polygon resolution, default 500) |
| `W_connect` | — | Width of the connecting straight channel [µm] |
| `L_taper` | — | Length of linear taper between W_connect and L₀ [µm] |

From the paper (Fig. 1a caption): the example uses σ̃L₀² = 0.23 and ω̃L₀² = 0.46 with L₀ = 100 µm.
Note that in the paper σ̃ and ω̃ are already normalised by L₀², so the input values are σ̃ and ω̃ directly.

---

## Installation

### Macro (standalone)
1. Copy `rheofluidics_macro.lym` to `~/.klayout/macros/` (Linux/macOS) or `%APPDATA%\klayout\macros\` (Windows)
2. In KLayout: **Macros → Macro Development** or find it in **Macros** menu → *Rheofluidics*
3. Or load it directly via **Macros → Import**

### PCell library (autorun)
1. Copy `rheofluidics_pcell_lib.lym` to `~/.klayout/macros/`
2. Restart KLayout
3. The library **Rheofluidics** appears in the **Libraries** panel on the left
4. Drag `RheoChannel_Constant` or `RheoChannel_Chirp` into your layout and edit parameters in the PCell dialog

---

## Modes

### Constant
Single sinusoidal stress segment: uniform σ̃ and ω̃ over N periods.  
→ Use `RheoChannel_Constant` or the macro in Constant mode.

### Chirp
Multiple segments concatenated, each with its own (σ̃ᵢ, ω̃ᵢ), yielding a frequency-swept stress profile as in Fig. 1e of the paper.  
→ Use `RheoChannel_Chirp` or the macro in Chirp mode.  
→ Provide comma-separated lists of equal length for σ̃ and ω̃.

---

## Geometry

The polygon is built as follows:
```
W_connect/2 ──┐                                    ┌── W_connect/2
               \                                  /
                └──[taper in]──[channel]──[taper out]──┘
               /                                  \
W_connect/2 ──┘                                    └── W_connect/2
```

- **Channel body**: 2×npts vertices from the numerical solution of eq.(1)
- **Tapers**: 2×30 vertices, linear interpolation between W_connect/2 and L₀
- **Symmetry**: the profile is symmetric about the channel axis (y=0)
- The polygon is centered at x=0 (taper inlet), extending to x = channel_length + L_taper

---

## Numerical method

The dimensionless ODE:

    dL̃/dt̃ = L̃ · σ̃ · sin(t̃),   L̃(0) = 1

is solved by RK4 over [0, 2π·N]. The spatial coordinate x̃ is recovered by trapezoidal integration of dx̃ = dt̃/L̃. The result is rescaled to the target `channel_length`.

For the chirp, segments are concatenated with relative x-stretching set by the ratio of ω̃ values between segments, preserving the correct time-to-space mapping for each frequency.
