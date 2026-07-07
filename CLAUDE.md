# AAI repo notes

## Docs site (Sphinx, `docs/`)

- **Never commit anything under `docs/build/`.** It's generated output
  (`sphinx-build` writes `.doctrees` cache + `html/` there) and is gitignored.
  If it ever shows up as tracked or staged again, `git rm -r --cached docs/build`
  and stop — don't commit it "just this once."

  Why this matters: a stale committed `docs/build/doctrees` cache makes builds
  incremental against a snapshot from whenever it was last committed instead of
  a clean build from current source. Sphinx's sidebar nav is assembled from the
  live environment at write time, but if the cached environment fails to load
  cleanly (e.g. after a myst-nb upgrade) some pages get skipped/rewritten
  inconsistently — new pages get the current nav, old unchanged pages can keep
  a stale one. Symptom looks exactly like "page A doesn't link to new page B,
  but page B links to A." A clean build never has this problem.

- **Before committing any change under `docs/source/` that touches a toctree**
  (new page, new chapter, renamed file, edited `main.md`/`overviews.md`/
  `index.md`), rebuild clean and check the nav from both directions:

  ```
  rm -rf docs/build && (cd docs && make html)
  ```

  Then grep the built sidebar on at least one old page and the new page to
  confirm each links to the other:

  ```
  grep -o 'href="[^"]*">[A-Za-z0-9 :]*$' docs/build/html/<old-page>.html
  grep -o 'href="[^"]*">[A-Za-z0-9 :]*$' docs/build/html/<new-page>.html
  ```

  Don't trust an incremental `make html` alone to validate nav changes — always
  confirm with a clean build first.

- `docs/requirements.txt` should keep every package pinned to an exact version
  (`==`), no bare/floating package names. An unpinned package silently
  installing a newer major version between builds is what breaks the cached
  environment above, and separately can break the build outright (see the
  `pandas==2.0.3` → no Python 3.12 wheel issue fixed in this repo already).
