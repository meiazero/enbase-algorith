# EnBaSe Replication

## Commands to build the examples and PDF

- Build the Python example using the `uv` tool:

```sh
uv run python examples/enbase_mnist.py
```

or

```sh
uv run python examples/spatial_enbase_mnist.py
```

- Build the Zig example:

```sh
zig build-exe examples/enbase_mnist.zig -O ReleaseSafe
```

or

```sh
zig build-exe examples/spatial_enbase_mnist.zig -O ReleaseSafe
```

- Build the LaTeX document using `latexmk` with `lualatex`:

```sh
latexmk -pdf -pdflatex="lualatex %O %S" -shell-escape -outdir=.github/docs enbase.tex
```

This project is licensed under the [GNU GPLv3 License](./LICENSE).
