# cellpose-gaussian

A Python package for Cellpose with Gaussian processing.

## Installation

### From source

```bash
git clone https://github.com/thjimmylee/cellpose-gaussian.git
cd cellpose-gaussian
pip install -e .
```

### For development

```bash
pip install -e ".[dev]"
```

Or install development dependencies separately:

```bash
pip install -r requirements-dev.txt
```

## Usage

```python
import cellpose_gaussian

# Your code here
```

## Development

### Running tests

```bash
pytest
```

### Code formatting

```bash
black src/ tests/
```

### Linting

```bash
flake8 src/ tests/
```

## Project Structure

```
cellpose-gaussian/
├── src/
│   └── cellpose_gaussian/
│       └── __init__.py
├── tests/
│   └── __init__.py
├── README.md
├── LICENSE
├── pyproject.toml
├── setup.py
├── setup.cfg
├── requirements.txt
├── requirements-dev.txt
└── MANIFEST.in
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.