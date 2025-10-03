## jaxhps/docs

Here is the documentation for jaxhps.

To build the docs locally, make sure you have the development requirements installed:
```
cd jaxhps
pip install -e .[dev]
```
Then move to the `docs` directory and run a `make` command:
```
cd docs
make html
```
This will build the docs in `_build/html`. The front page of the docs will be located at `_build/html/index.html`. 