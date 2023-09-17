I've chosen to work with pytest for testing on this project, its clean and less verbose syntax compared to the traditional unittest made me want do dive deeper into the library.

The project boasts a test coverage of 98% using pytest.

At present, the test modules lack comprehensive documentation, an issue I plan to address in the near future. As it stands, the existing documentation structure is as follows:

- Each section of the test file is marked with comments that specify the class and methods under test.

- The function names within these sections have descriptive titles, detailing the specific aspects of each method being tested.

- If no specific aspects are being tested, the function name is simply the name of the method being tested.

- More involved tests will be accompanied with additional comments

For example:

```python
# TESTING GridSearch CLASS

# Test for the 'execute' method
def test_execute(setup_grid):
    setup_grid.process_model = Mock()
    assert setup_grid.execute() is None
    assert setup_grid.process_model.call_count == 1

# Test for executing two models
def test_execute_two_models(setup_two_models):
    setup_two_models.process_model = Mock()
    assert setup_two_models.execute() is None
    assert setup_two_models.process_model.call_count == 2
```
