# Copilot Instructions for napari-pyvertexmodel

## Project Overview

**napari-pyvertexmodel** is a napari plugin that enables running 3D mechanical simulations with pyVertexModel. This plugin provides an interactive GUI within napari for configuring and running vertex-based tissue mechanics simulations, visualizing results as 3D surface meshes.

### Key Technologies
- **Framework**: napari plugin (npe2 manifest-based)
- **Language**: Python 3.10 - 3.13
- **GUI Framework**: magicgui for widget creation, qtpy for Qt bindings
- **Core Dependencies**: pyVertexModel (simulation engine), VTK (3D visualization), scikit-image
- **Package Management**: pip, uv (in CI), setuptools with setuptools_scm for versioning
- **Testing**: pytest with pytest-qt, pytest-cov
- **Build System**: setuptools (PEP 517 compliant)

## Repository Structure

```
napari-pyvertexmodel/
├── .github/
│   ├── workflows/test_and_deploy.yml  # CI/CD pipeline
│   ├── ISSUE_TEMPLATE/                # Bug reports, feature requests
│   └── PULL_REQUEST_TEMPLATE.md
├── src/napari_pyvertexmodel/          # Source code (src-layout)
│   ├── __init__.py                    # Plugin exports
│   ├── napari.yaml                    # Napari plugin manifest (npe2)
│   ├── _widget.py                     # Main widget (Run3dVertexModel)
│   ├── _reader.py                     # File readers (.npy, .pkl)
│   ├── _writer.py                     # File writers
│   ├── _sample_data.py                # Sample data provider
│   └── utils.py                       # VTK/napari visualization utilities
├── tests/                             # Test suite
│   ├── test_widget.py
│   ├── test_reader.py
│   ├── test_writer.py
│   └── test_sample_data.py
├── pyproject.toml                     # Project metadata and configuration
├── tox.ini                            # Tox configuration for multi-env testing
├── .pre-commit-config.yaml            # Pre-commit hooks
├── .gitignore
└── README.md
```

### Important Files to Know
- **napari.yaml**: Defines plugin contributions (widgets, readers, writers, sample data)
- **_widget.py**: Main entry point for the GUI widget with mechanical simulation parameters
- **utils.py**: Contains mesh creation and napari layer management functions
- **pyproject.toml**: All project configuration including dependencies, build settings, tool configs

## Development Workflow

### Installation

**Standard installation:**
```bash
pip install -e .
```

**With development dependencies (using dependency groups):**
```bash
pip install -e . --config-settings editable_mode=compat
pip install tox-uv pytest pytest-cov pytest-qt "napari[qt]"
```

**Note**: The project uses PEP 735 dependency groups (`[dependency-groups]` in pyproject.toml) rather than optional dependencies for dev tools. These are not automatically installed with `-e ".[dev]"`.

**With napari and Qt backend:**
```bash
pip install "napari-pyvertexmodel[all]"
```

### Running Tests

**Using pytest directly (fastest for development):**
```bash
pytest -v
pytest tests/test_widget.py  # Run specific test file
pytest -v --cov=napari_pyvertexmodel --cov-report=xml  # With coverage
```

**Using tox (runs full test matrix):**
```bash
tox  # Runs all environments
tox -e py312-linux  # Run specific environment
```

**Note**: Tests require a Qt backend and may need display configuration. CI uses `pyvista/setup-headless-display-action@v4.2` for headless testing.

### Linting and Formatting

The project uses:
- **black** for code formatting (line length: 79)
- **ruff** for linting (includes flake8, pyupgrade, isort rules)
- **pre-commit hooks** for automatic checks

**Run linters:**
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files

# Or run tools directly
black src/ tests/
ruff check src/ tests/
ruff check --fix src/ tests/  # Auto-fix issues
```

**Important ruff configuration:**
- Ignores `E501` (line length - handled by black)
- Ignores `UP006`, `UP007` (type annotations - required for magicgui runtime annotations)

### Building the Package

```bash
python -m build  # Creates source distribution and wheel in dist/
```

### Version Management

- Uses **setuptools_scm** for automatic versioning from git tags
- Version written to `src/napari_pyvertexmodel/_version.py` (auto-generated, gitignored)
- Fallback version: `0.0.1+nogit` if not in git repo

## Code Guidelines

### General Principles
1. **Minimal changes**: Make surgical, focused modifications
2. **Test coverage**: Maintain or improve existing test coverage
3. **Code style**: Follow existing patterns, use black formatting, comply with ruff rules
4. **Type hints**: Use runtime type annotations compatible with magicgui (e.g., `"napari.layers.Image"`)
5. **Error handling**: Use broad exception catching (`except Exception`) with `# noqa: BLE001` only where appropriate

### Key Patterns in This Codebase

**1. napari Plugin Architecture (npe2 manifest)**
- Plugin contributions defined in `napari.yaml`
- Commands reference Python callables (e.g., `napari_pyvertexmodel._reader:napari_get_reader`)
- Widgets use magicgui `Container` class

**2. magicgui Widgets**
- Create widgets using `create_widget()` with `annotation` and `widget_type`
- Container-based layout (`.extend()`, `.append()`, `.remove()`)
- Connect callbacks with `.clicked.connect()`
- Example:
```python
self._slider = create_widget(
    label="Parameter",
    annotation=float,
    widget_type="FloatSlider"
)
```

**3. VTK to napari Conversion**
- VTK polydata converted to napari surface layers
- Use `vtk_to_numpy()` for vertices, faces, scalars
- Batch multiple cells into single layer for performance
- See `utils.py:_add_surface_layer()` and `_get_mesh()`

**4. File I/O**
- `.pkl` files: pyVertexModel simulation states (reader in `_reader.py:pkl_reader_function`)
- `.npy` files: Numpy arrays for label images (reader in `_reader.py:npy_reader_function`)
- Reader plugins return list of tuples: `[(data, add_kwargs, layer_type)]`

### Common Pitfalls

1. **Display configuration for tests**: Tests involving napari require a display backend. In CI, use headless display setup. Locally, ensure `QT_QPA_PLATFORM` is set if needed.

2. **Dependency group syntax**: The project uses `[dependency-groups]` (PEP 735), not `[project.optional-dependencies]` for dev dependencies. Install manually or via tox.

3. **Type annotations for magicgui**: Don't use `from __future__ import annotations` or PEP 604 union syntax (`X | Y`) in files with magicgui widgets, as magicgui requires runtime type evaluation.

4. **VTK version compatibility**: VTK dependency comes from pyVertexModel. Be aware of potential compatibility issues with different VTK versions.

5. **Temporary directories**: The widget creates temporary directories for simulation output. Ensure proper cleanup in `__del__` method.

6. **setuptools_scm**: Don't manually edit `_version.py` - it's auto-generated. Version changes require git tags.

## Testing

### Test Structure
- Tests use pytest with pytest-qt for widget testing
- Coverage target: maintain or improve existing coverage
- Tests are organized by functionality (reader, writer, widget, sample_data)

### Writing Tests for napari Widgets
```python
import numpy as np
from napari_pyvertexmodel._widget import Run3dVertexModel

def test_widget(make_napari_viewer):
    """Test widget initialization."""
    viewer = make_napari_viewer()
    widget = Run3dVertexModel(viewer)
    # Test widget properties
    assert widget._viewer == viewer
```

**Note**: Use the `make_napari_viewer` fixture provided by napari’s pytest integration (with pytest-qt used alongside it for Qt event handling), rather than creating a viewer manually.

### Running Specific Tests
```bash
pytest tests/test_widget.py::test_specific_function -v
pytest -k "test_reader"  # Run tests matching pattern
pytest -x  # Stop on first failure
```

## CI/CD Pipeline

### GitHub Actions Workflow (`.github/workflows/test_and_deploy.yml`)

**Test matrix:**
- Platforms: ubuntu-latest, windows-latest, macos-latest
- Python versions: 3.10, 3.11, 3.12, 3.13
- Runs on: push to main/npe2, pull requests, tags (v*)

**Key steps:**
1. Setup Python with `astral-sh/setup-uv@v7`
2. Install headless display for GUI testing (pyvista action)
3. Run tests with tox via `uvx --with tox-gh-actions tox`
4. Upload coverage to codecov

**Deployment:**
- Triggered by version tags (v*)
- Builds package with `hynek/build-and-inspect-python-package@v2`
- Publishes to PyPI using trusted publisher (OIDC)

### Debugging CI Failures

**Common issues:**
1. **Display/Qt errors**: Ensure headless display is properly configured
2. **Dependency resolution**: Check `uv.lock` is up to date
3. **Platform-specific failures**: Test locally with tox platform filter
4. **Import errors**: Verify napari.yaml command paths match actual module structure

## Making Changes

### Adding a New Widget Parameter

1. **Add slider/widget in `_widget.py:__init__`:**
```python
self._new_param_slider = create_widget(
    label="New Param",
    annotation=float,
    widget_type="FloatSlider"
)
self._new_param_slider.value = 1.0
```

2. **Extend container:**
```python
self.extend([..., self._new_param_slider])
```

3. **Update model sync methods:**
```python
# In _update_sliders_from_model:
self._new_param_slider.value = self.v_model.set.NewParam

# In _update_model_from_sliders:
self.v_model.set.NewParam = self._new_param_slider.value
```

4. **Add tests in `tests/test_widget.py`**

### Adding a New File Format Reader

1. **Update `_reader.py:napari_get_reader`** to recognize extension
2. **Create reader function** returning `[(data, add_kwargs, layer_type)]`
3. **Update napari.yaml** to register new file patterns
4. **Add tests in `tests/test_reader.py`**

### Modifying Visualization

- Edit `utils.py` functions: `_add_surface_layer`, `_get_mesh`, `_create_surface_data`
- VTK data conversion is sensitive - test thoroughly with real simulation data
- Performance matters: batch operations where possible

## Troubleshooting

### Issue: "ImportError: cannot import name '_version'"
**Solution**: The package hasn't been installed in editable mode. Run `pip install -e .`

### Issue: "No Qt bindings could be found"
**Solution**: Install Qt backend: `pip install "napari[all]"` or `pip install PyQt5`

### Issue: "WARNING: napari-pyvertexmodel does not provide the extra 'dev'"
**Solution**: This is expected. Install dev dependencies manually:
```bash
pip install tox-uv pytest pytest-cov pytest-qt
```
**Root cause**: The project uses PEP 735 dependency groups (`[dependency-groups]`) instead of optional dependencies (`[project.optional-dependencies]`). This is a newer standard that pip doesn't yet fully support with the `.[dev]` syntax.

### Issue: Tests fail with display errors
**Solution**:
- For CI: Ensure headless display action is configured
- For local: Set `QT_QPA_PLATFORM=offscreen` or install Xvfb

### Issue: "ModuleNotFoundError: No module named 'pyVertexModel'"
**Solution**: pyVertexModel is a required dependency. Ensure it installs correctly:
```bash
pip install pyvertexmodel
```

### Issue: `uv` command not found locally
**Solution**: `uv` is used in CI but not required for local development. Use standard `pip` for local work:
```bash
pip install -e .
pytest
```
If you want to use `uv` locally: `pip install uv` or follow https://github.com/astral-sh/uv

## Errors Encountered During Repository Exploration

During the creation of this guide, the following issues were observed:

1. **Dev dependencies not installable via `.[dev]`**:
   - Error: `WARNING: napari-pyvertexmodel does not provide the extra 'dev'`
   - Cause: Project uses PEP 735 `[dependency-groups]` which is not yet fully supported by pip's extras syntax
   - Workaround: Install dev dependencies manually or use tox

2. **`uv` not available in local environment**:
   - The CI pipeline uses `uv` for faster dependency resolution, but it's not a hard requirement
   - Local development works fine with standard `pip`
   - No installation errors encountered when using pip

## Additional Resources

- **napari plugin development**: https://napari.org/stable/plugins/
- **npe2 manifest specification**: https://napari.org/stable/plugins/technical_references/manifest.html
- **magicgui documentation**: https://pyapp-kit.github.io/magicgui/
- **pyVertexModel documentation**: https://github.com/pablo1990/pyVertexModel

## Workflow Summary for Agents

When working on this repository:

1. **Before making changes:**
   - Understand the napari plugin architecture and magicgui patterns
   - Check existing test coverage for the area you're modifying
   - Review the specific module's structure and patterns

2. **Development cycle:**
   - Make minimal, focused changes
   - Run `black` and `ruff` after edits
   - Write/update tests for your changes
   - Run relevant tests: `pytest tests/test_<module>.py -v`
   - Run full test suite before finalizing

3. **Before submitting:**
   - Run `pre-commit run --all-files`
   - Ensure all tests pass: `pytest -v --cov=napari_pyvertexmodel`
   - Check that coverage is maintained
   - Verify package builds: `python -m build`

4. **Common modifications:**
   - **Widget changes**: Edit `_widget.py`, update tests
   - **File I/O**: Edit `_reader.py` or `_writer.py`, update `napari.yaml`
   - **Visualization**: Edit `utils.py` (VTK/napari layer functions)
   - **Dependencies**: Update `pyproject.toml` `dependencies` list

---

*This file is intended to help coding agents quickly understand and work effectively with this repository. Keep it updated as the project evolves.*
