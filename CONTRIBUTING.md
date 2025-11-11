# Contributing to RFT v2 Galaxy Rotations

Thank you for your interest in contributing to the RFT v2 galaxy rotation curves project!

## Quick Start

To reproduce the results locally:

```bash
./RUNME.sh
```

This script will:
- Set up the conda environment (or use Docker)
- Run the full TEST cohort
- Verify baseline and hash locks
- Generate comparison reports

## Development Workflow

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR-USERNAME/rft-v2-galaxy-rotations.git
cd rft-v2-galaxy-rotations
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

- Follow existing code style (we use Black for Python)
- Add tests for new functionality
- Update documentation as needed

### 4. Run Tests Locally

```bash
# Run baseline lock check
python scripts/audit_baselines.py

# Run hash lock check
python scripts/verify_final_numbers_hash.py

# Run any unit tests
pytest
```

### 5. Commit and Push

```bash
git add .
git commit -m "Brief description of changes"
git push origin feature/your-feature-name
```

### 6. Open a Pull Request

- Go to the repository on GitHub
- Click "New Pull Request"
- Select your branch
- Fill out the PR template
- Wait for CI checks to pass

## Code Style

- **Python**: Follow PEP 8, use Black for formatting
- **Line length**: 100 characters maximum
- **Imports**: Group standard library, third-party, and local imports
- **Docstrings**: Use Google style

## Testing Requirements

All PRs must pass:
- ✅ Baseline lock (frozen numbers unchanged)
- ✅ Hash lock (final_numbers.json integrity)
- ✅ Paper build (LaTeX compiles successfully)
- ✅ Any existing unit tests

## Frozen Results

**Important**: The `results/` directory contains frozen validation outputs. Changes to these files will fail CI unless explicitly approved by maintainers. If you believe results need updating:

1. Open an issue explaining why
2. Get approval before submitting PR
3. Update hash locks accordingly

## Issue Templates

Use the provided issue templates:
- 🐛 **Bug Report**: For reproducibility issues or errors
- ✨ **Feature Request**: For new analysis ideas
- 📝 **Documentation**: For doc improvements

## Scientific Contributions

If you're proposing changes to:
- **Methodology**: Open an issue first to discuss
- **Baselines**: Must include justification and comparison
- **Parameters**: Must preserve k=0 constraint for fair comparison

## Questions?

- Open an issue for questions about the methodology
- Check existing issues for common questions
- See README.md for quick-start guide

## Code of Conduct

Be respectful, professional, and constructive. We're all here to advance science.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
