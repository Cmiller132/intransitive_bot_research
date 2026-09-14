# Agent rules

- Everything added here has a purpose. One-time scripts and documents are
  deleted after use. Anything used often becomes one robust, reusable tool.
- One solution per problem. Improve or replace the existing one; never add a
  second.
- Every line of code must have a reason. Remove unused features completely. No
  legacy support, no fallbacks.
- Only the documents listed in Project.md exist. No proposals, history,
  results logs or session notes.
- Comments exist to make the codebase easier to understand: what a module,
  class or function does and what it interacts with. They do not explain why
  something is there or narrate history. Keep them short; delete any that
  stop being accurate.
- Game rules live only in the engine crate.
- Strength is measured only by the eval tool.
- Build only user-approved design items; ask before changing an approved one.
- One GPU training run at a time. Check before launching; never interrupt a
  run except for a failure.
- Deploying to the site is the user's decision.
- Before ending a session, run checks relevant to the changes. Run a model's
  CPU tests only when that model or its dependencies changed; use
  `cargo xtask check` when full-workspace validation is relevant.
- Before ending a session: delete your scratch files and diagnostics, remove
  code and comments your changes made obsolete, update any README whose
  interface changed, and report plainly what was measured and what is undone.
