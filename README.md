# SEMAT Project

This is a Python project using a modern project structure.

## Project Structure

```
semat/
│
├── src/               # Source code
├── tests/             # Test files
├── requirements.txt   # Project dependencies
└── setup.py          # Package and distribution management
```

## Getting Started

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- On Windows:
```bash
.\venv\Scripts\activate
```
- On Unix or MacOS:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Development

- Add your source code in the `src` directory
- Write tests in the `tests` directory
- Add any project dependencies to `requirements.txt`

## Testing

To run tests (once you've added them):
```bash
python -m pytest tests/
```