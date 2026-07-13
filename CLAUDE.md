# AAI repo notes

## Docs site (Sphinx, `docs/`)

- **Never commit anything under `docs/build/`.** It's generated output
  (`sphinx-build` writes `.doctrees` cache + `html/` there) and is gitignored.
  If it ever shows up as tracked or staged again, `git rm -r --cached docs/build`
  and stop — don't commit it "just this once."

  Why this matters: a stale committed `docs/build/doctrees` cache makes builds
  incremental against a snapshot from whenever it was last committed instead of
  a clean build from current source. When that cached environment loads
  inconsistently (e.g. after a dependency upgrade), some pages get rewritten
  and others keep a stale sidebar — the symptom is "page A doesn't link to new
  page B, but B links to A." A clean build never has this problem. (This
  happened once already; the fix was reverted by an unrelated merge and had to
  be re-applied — check for it after big merges.)

- **Before committing any change under `docs/source/` that touches a toctree**
  (new page, new chapter, renamed file, edited `main.md`/`overviews.md`/
  `index.md`), rebuild clean and check the nav from both directions:

  ```
  rm -rf docs/build && (cd docs && make html)
  ```

  Then grep the built sidebar on one old page and the new page to confirm each
  links to the other:

  ```
  grep -o 'href="[^"]*">[A-Za-z0-9 :]*$' docs/build/html/<old-page>.html
  grep -o 'href="[^"]*">[A-Za-z0-9 :]*$' docs/build/html/<new-page>.html
  ```

- In MyST toctree/directive options, never end an option value with a colon
  (`:caption: Chapters:` is invalid YAML → "Invalid options format" warning and
  the options are silently dropped). Use `:caption: Chapters`.

- `docs/requirements.txt` should keep every package pinned to an exact version
  (`==`), no bare/floating package names. An unpinned package silently
  installing a newer major version between builds is what breaks the cached
  environment above, and can break the build outright (pandas 2.0.3 had no
  Python 3.12 wheel; myst_nb was unpinned).

- Display math must be a standalone block with **blank lines before and
  after** the `$$` fences. A `$$ ... $$` equation crammed directly against a
  prose line (no blank line between) is not parsed as display by MyST
  dollarmath (`dmath_double_inline` is off) — the `$` pairing cascades and
  renders the surrounding text as run-on italic math. Write:

  ```
  text before,

  $$
  equation
  $$

  more text.
  ```

- Hidden exercise solutions in lessons use native HTML `<details>` /
  `<summary>Solution</summary>` blocks with blank lines around the inner
  markdown (so MyST parses the math inside). No extra Sphinx extension needed.
