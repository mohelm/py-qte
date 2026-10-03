# Releasing

Releases are published to [PyPI](https://pypi.org/project/py-qte/) by the
[`Release`](.github/workflows/release.yml) GitHub Actions workflow, using PyPI trusted publishing
(no stored token). The documentation website is published separately — see
[`docs/README.md`](docs/README.md).

## Versioning

The version is declared in **two** files and both must match:

- `pyproject.toml` (`version = "..."`), and
- `meson.build` (`project('py-qte', 'c', version: '...')`).

The version follows PEP 440 (e.g. `0.1.0a1` for an alpha). Check that the two agree with:

```sh
uv run python tools/check_versions.py
```

`tools/check_versions.py` is wired into the pre-commit hooks and the release workflow; in the
workflow it also checks the tag (`v<version>`) against the declared version.

## Cutting a release

1. **Bump the version** in `pyproject.toml` and `meson.build`, then verify:

   ```sh
   uv run python tools/check_versions.py
   ```

2. **Regenerate the PyPI long description.** `project.readme` points at `README_PYPI.md`, which is
   generated from `README.md` with relative links made absolute:

   ```sh
   uv run python -m docs.readme --pypi-readme-only
   ```

3. **Update the docs website** if the public API or the README changed — see
   [`docs/README.md`](docs/README.md).

4. **Commit** the version bump and the regenerated files on `main`.

5. **(Optional) dry run on TestPyPI.** Open *Actions → Release → Run workflow* and leave the
   `testpypi` input checked. This builds every artifact and publishes to
   [test.pypi.org/project/py-qte](https://test.pypi.org/project/py-qte) without touching PyPI.

6. **Tag and publish.**

   ```sh
   git tag v0.1.0a2
   git push origin v0.1.0a2
   ```

   The workflow verifies the tag, then builds and uploads the sdist, a pure fallback wheel
   (`FC=/nonexistent-gfortran`, for platforms without a prebuilt extension), and cibuildwheel
   wheels for `ubuntu-latest`, `macos-15-intel`, and `macos-15`.

7. **Verify** the [PyPI release](https://pypi.org/project/py-qte/) and, if applicable, the website.

## Notes

- The tag must be `v<version>` (with the `v` prefix) and match `pyproject.toml`; the workflow
  fails otherwise.
- There is currently no `CHANGELOG` file.
