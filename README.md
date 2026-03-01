# wasp-differentiated-mujoco-iros2026

**This repository is built upon the [mujoco](https://github.com/google-deepmind/mujoco) codebase. We thank the authors for open-sourcing their work.**

## Role in the Main Project

For the main project `mujoco_wasp_mpc`, this repository serves as a low-level supporting/complementary backend.

It intentionally preserves the original MuJoCo project structure so that the main project can compile cleanly by substituting the anonymous placeholder links/tags with this repository link, as described in the `mujoco_wasp_mpc` README.

## Core Contribution

The primary implementation effort in this repository is concentrated in:

- `src/engine/engine_derivative_wasp.h`
- `src/engine/engine_derivative_wasp.c`

These files implement WASP-based model derivative routines at the simulator engine level.

## Design Principle: FD-Compatible Structure, WASP Acceleration

The WASP derivative implementation is written to mirror the structure of `engine_derivative_fd` as closely as possible, while replacing the finite-difference strategy with a faster WASP method.

This includes explicit decomposition and handling of derivative channels such as:

- `Dq`
- `Dv`
- `Da`
- `Du`

Maintaining this structural alignment provides two key benefits:

- engineering continuity with MuJoCo’s established derivative pipeline
- low-friction integration into higher-level MPC/planner code that already expects FD-like derivative semantics

## Engineering Character

This repository focuses on a difficult systems task: upgrading derivative performance without breaking engine-level interfaces or architectural boundaries.

The implementation aims for an elegant balance:

- strict compatibility with existing derivative organization patterns
- practical performance-oriented WASP internals
- clean separation between low-level simulator differentiation and upper-layer planner logic


