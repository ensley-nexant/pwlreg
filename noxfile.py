"""Nox file for pwlreg."""
import sys
from pathlib import Path
from typing import Iterable, Iterator

import nox
from nox import Session

python_versions = ["3.11", "3.10"]
nox.options.sessions = (
    "pre-commit",
    "tests",
    "docs-build",
)


def install(session: nox.Session, *, groups: Iterable[str], root: bool = True) -> None:
    """Install the dependency groups using Poetry.

    This function installs the given dependency groups into the session's
    virtual environment. When ``root`` is true (the default), the function
    also installs the root package and its default dependencies.

    To avoid an editable install, the root package is not installed using
    ``poetry install``. Instead, the function invokes ``pip install .``
    to perform a PEP 517 build.

    Args:
        session: The Session object.
        groups: The dependency groups to install.
        root: Install the root package.
    """
    session.run_always(
        "poetry",
        "install",
        "--no-root",
        "--sync",
        "--{}={}".format("with" if root else "only", ",".join(groups)),
        external=True,
    )
    if root:
        session.install(".")


def export_requirements(session: nox.Session, *, extras: Iterable[str] = ()) -> Path:
    """Export a requirements file from Poetry.

    This function uses ``poetry export`` to generate a requirements file
    containing the default dependencies at the versions specified in
    ``poetry.lock``.

    Args:
        session: The Session object.
        extras: Extras supported by the project.

    Returns:
        The path to the requirements file.
    """
    output = session.run_always(
        "poetry",
        "export",
        "--format=requirements.txt",
        "--without-hashes",
        *[f"--extras={extra}" for extra in extras],
        external=True,
        silent=True,
        stderr=None,
    )

    if output is None:
        session.skip(
            "The command `poetry export` was not executed "
            "(a possible cause is specifying `--no-install`)"
        )

    assert isinstance(output, str)  # noqa: S101

    def _stripwarnings(lines: Iterable[str]) -> Iterator[str]:
        for line in lines:
            if line.startswith("Warning:"):
                print(line, file=sys.stderr)
                continue
            yield line

    text = "".join(_stripwarnings(output.splitlines(keepends=True)))

    path = session.cache_dir / "requirements.txt"
    path.write_text(text)

    return path


@nox.session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite."""
    install(session, groups=["coverage", "tests"])
    try:
        session.run("coverage", "run", "--parallel", "-m", "pytest", *session.posargs)
    finally:
        if session.interactive:
            session.notify("coverage", posargs=[])


@nox.session(python=python_versions)
def coverage(session):
    """Generate coverage data."""
    args = session.posargs or ["report"]

    install(session, groups=["coverage"], root=False)

    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", *args)


@nox.session(name="pre-commit", python=python_versions)
def pre_commit(session: Session) -> None:
    """Lint using pre-commit."""
    args = session.posargs or ["run", "--all-files", "--show-diff-on-failure"]
    install(session, groups=["pre-commit"], root=False)
    session.run("pre-commit", *args)


@nox.session(name="docs-build", python=python_versions[0])
def docs_build(session: Session) -> None:
    """Build the docs with mkdocs."""
    args = session.posargs
    install(session, groups=["docs"], root=True)
    session.run("mkdocs", "build", *args)


@nox.session(name="docs-deploy", python=python_versions[0])
def docs_deploy(session: Session) -> None:
    """Build the docs with mkdocs."""
    args = session.posargs
    install(session, groups=["docs"], root=True)
    session.run("mkdocs", "gh-deploy", *args)
