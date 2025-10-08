# Hessian Approximation for Toy Models

## Getting Started

Using `uv` as package manager, therefore assuming as a first step, that uv is installed

### Install uv

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup

```bash
git clone <repo>
cd <repo>
uv sync
```

### Run

```bash
uv run python main.py
```

## Development

### Adding Dependencies

```bash
# Add a runtime dependency
uv add requests

# Add a development dependency
uv add --dev pytest
```

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black .
```

### Linting

```bash
uv run ruff check .
```

## Project Structure

```
my-project/
├── .python-version    # Python version (3.11)
├── pyproject.toml     # Project configuration and dependencies
├── README.md          # This file
├── main.py           # Main application file
└── .venv/            # Virtual environment (auto-generated)
```

## Common Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install/update all dependencies |
| `uv add <package>` | Add a new dependency |
| `uv remove <package>` | Remove a dependency |
| `uv run <command>` | Run a command in the virtual environment |
| `uv python list` | List available Python versions |

## Troubleshooting

### Python Version Issues

If you encounter Python version conflicts, ensure you're using Python 3.11:

```bash
uv python install 3.11
uv python pin 3.11
uv sync
```

### Virtual Environment

uv manages the virtual environment automatically. If you need to reset it:

```bash
rm -rf .venv
uv sync
```

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.