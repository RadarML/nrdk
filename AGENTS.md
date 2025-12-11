# AGENTS.md: AI Agent Guide for NRDK

This document provides guidance for AI agents working with the Neural Radar Development Kit (NRDK), a Python framework for developing, training, and evaluating machine learning models on radar spectrum and multimodal sensor data.

## Project Architecture Overview

The NRDK is built around **modular components** that can be assembled via dependency injection using Hydra configuration. The framework follows a strict separation of concerns:

- **Core Framework**: PyTorch Lightning-based training pipeline
- **Data Loading**: Abstract dataloader specifications for modular data processing
- **Training Objectives**: Standardized loss functions and metrics
- **Model Zoo**: Reusable, stable model architectures and modules

## Key Design Principles

### Modularity
Components should implement well-defined interfaces and be composable. Use the abstract dataloader specifications for data processing and the objective interface for training losses.

### Type Safety
All code should be fully typed using `jaxtyping` and `beartype`. Type annotations are both statically checked and enforced at runtime.

### Configuration-Driven
Use Hydra configuration files for all experiments. Configuration should be hierarchical and composable.

### Stability Focus
Only merge stable, production-ready code into the main repository. Research code should live in separate repositories until proven stable.

## Adding New Components

### Data Processing Components

When adding new data preprocessing:

1. **Implement Abstract Interfaces**: Use `abstract_dataloader.spec` interfaces (`Sensor`, `Trace`, `Dataset`, `Transform`, `Collate`)
2. **Protocol Types**: Leverage protocol types for compatibility with existing transforms
3. **Configuration**: Create corresponding Hydra config files in the appropriate config group
4. **Documentation**: Include type specifications for inputs and outputs

### Model Architectures

When adding new models:

1. **Extend Standard Modules**: Build on `torch.nn.Module` or use `TokenizerEncoderDecoder` wrapper
2. **Size Configuration**: Reference global size parameters from the `size` config group
3. **Component Structure**: Organize as tokenizer/encoder/decoder components when applicable
4. **Type Annotations**: Specify exact tensor shapes using `jaxtyping`

### Training Objectives

When implementing new objectives:

1. **Follow Objective Interface**: Implement `abstract_dataloader.ext.objective.Objective`
2. **Return Loss and Metrics**: Return scalar loss and dictionary of metrics
3. **Type Ground Truth**: Specify expected input/output types clearly
4. **Support Multi-Objectives**: Ensure compatibility with `MultiObjective` wrapper

### CLI Tools

When extending CLI functionality:

1. **Use Tyro Framework**: Follow existing patterns with positional and flagged arguments
2. **Validation**: Include validation functions for configurations and data
3. **Export/Import**: Support standard export formats for interoperability
4. **Help Documentation**: Provide clear usage examples and parameter descriptions

## Testing and Validation

### Running Tests

NRDK uses pytest for automated testing. To run the test suite:

```bash
# Run all tests
uv run --extra dev pytest tests

# Run tests with coverage report
uv run --extra dev pytest --cov=src/nrdk tests

# Run specific test file
uv run --extra dev pytest tests/test_basic.py

# Run with verbose output
uv run --extra dev pytest -v tests
```

### Writing Tests

When contributing to NRDK, include tests for new functionality:

1. **Test Structure**: Place tests in the `tests/` directory
2. **Naming Convention**: Use `test_*.py` for test files and `test_*` for test functions
3. **Test Categories**:
   - Unit tests: Test individual components in isolation
   - Integration tests: Test component interactions
   - Configuration tests: Validate Hydra configuration loading

Example test structure:
```python
import pytest
import nrdk

def test_component_functionality():
    """Test basic component functionality."""
    # Test implementation
    assert True

class TestComponentClass:
    """Test class for comprehensive component testing."""
    
    def test_initialization(self):
        """Test component initialization."""
        # Test implementation
        pass
```

### Code Quality

- **Type Checking**: All code must pass strict type checking with `pyright`
- **Linting**: Code must pass `ruff` linting checks
- **Runtime Validation**: Use `beartype` for runtime type validation
- **Import Hooks**: Leverage jaxtyping import hooks for automatic validation
- **Test Coverage**: Maintain reasonable test coverage for critical components

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:

```bash
# Install pre-commit hooks
uv run --extra dev pre-commit install

# Run all hooks manually
uv run --extra dev pre-commit run --all-files
```

Hooks include:
- **ruff**: Code linting and formatting
- **pyright**: Static type checking
- **pytest**: Automated test execution

### Configuration Validation

Use the `nrdk validate` CLI tool to:
- Validate configuration file syntax
- Check parameter compatibility
- Verify data path existence
- Ensure objective/model alignment

### Data Validation

- **Sensor Data**: Validate sensor data formats and synchronization
- **Transform Pipelines**: Test data preprocessing chains
- **Batch Processing**: Verify collation and batching behavior

## Development Workflow

### Project Setup

1. **Clone/Submodule**: Add NRDK as submodule to your project
2. **Dependencies**: Install with appropriate extras (`roverd`, `grt`, `docs`, `dev`)
3. **Configuration**: Copy GRT reference configuration as starting point
4. **Virtual Environment**: Use `uv` for dependency management

### Experimentation

1. **Configuration First**: Create Hydra configs before implementing
2. **Incremental Development**: Build components incrementally with validation
3. **Logging**: Use rich logging with appropriate log levels
4. **Checkpointing**: Leverage PyTorch Lightning's automatic checkpointing

### Integration

1. **Interface Compliance**: Ensure components follow abstract specifications
2. **Configuration Testing**: Test configuration composition and overrides
3. **Performance Monitoring**: Use included performance monitoring callbacks
4. **Documentation**: Update type specifications and usage examples

### Configuration Structure

```yaml
# Standard config pattern
_target_: module.path.ClassName
param1: value1
param2: ${size.d_model}  # Reference global sizes
nested_config:
  _target_: another.module.Class
```

## Best Practices

### Type Safety
- Always use complete type annotations
- Leverage jaxtyping for tensor shapes
- Use beartype for runtime validation

### Configuration Management
- Keep configs small and focused
- Use composition over large monolithic configs
- Reference global parameters where appropriate

### Error Handling
- Provide clear error messages for configuration issues
- Validate inputs at component boundaries
- Use logging for debugging information

### Performance
- Use PyTorch Lightning's optimization features
- Implement efficient data loading patterns
- Monitor memory usage in long training runs

## Related Repositories

When extending NRDK functionality, consider these related projects:

- **red-rover**: Multimodal radar spectrum ecosystem
- **xwr**: Python interface for TI mmWave radars
- **abstract-dataloader**: Core data loading specifications

Keep these dependencies minimal and well-defined to maintain NRDK's modularity.
