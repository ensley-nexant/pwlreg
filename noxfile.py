"""Nox file for pwlreg."""
import os
import shlex
from pathlib import Path
from textwrap import dedent

import nox

try:
    from nox_poetry import Session, session
except ImportError:
    message = """\
    Nox failed to import the 'nox-poetry' package."""
    raise SystemExit(dedent(message)) from None

python_versions = ["3.11", "3.10"]
nox.options.sessions = ("pre-commit", "tests", "coverage")


def activate_virtualenv_in_precommit_hooks(session: Session) -> None:
    """Activate virtualenv in hooks installed by pre-commit.

    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.

    Args:
        session: The Session object.
    """
    assert session.bin is not None  # nosec noqa

    # Only patch hooks containing a reference to this session's bindir. Support
    # quoting rules for Python and bash, but strip the outermost quotes so we
    # can detect paths within the bindir, like <bindir>/python.
    bindirs = [
        bindir[1:-1] if bindir[0] in "'\"" else bindir
        for bindir in (repr(session.bin), shlex.quote(session.bin))
    ]

    virtualenv = session.env.get("VIRTUAL_ENV")
    if virtualenv is None:
        return

    headers = {
        # pre-commit < 2.16.0
        "python": f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {session.bin!r},
                os.environ.get("PATH", ""),
            ))
            """,
        # pre-commit >= 2.16.0
        "bash": f"""\
            VIRTUAL_ENV={shlex.quote(virtualenv)}
            PATH={shlex.quote(session.bin)}"{os.pathsep}$PATH"
            """,
        # pre-commit >= 2.17.0 on Windows forces sh shebang
        "/bin/sh": f"""\
            VIRTUAL_ENV={shlex.quote(virtualenv)}
            PATH={shlex.quote(session.bin)}"{os.pathsep}$PATH"
            """,
    }

    hookdir = Path(".git") / "hooks"
    if not hookdir.is_dir():
        return

    for hook in hookdir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        if not hook.read_bytes().startswith(b"#!"):
            continue

        text = hook.read_text()

        if not any(
            Path("A") == Path("a") and bindir.lower() in text.lower() or bindir in text
            for bindir in bindirs
        ):
            continue

        lines = text.splitlines()

        for executable, header in headers.items():
            if executable in lines[0].lower():
                lines.insert(1, dedent(header))
                hook.write_text("\n".join(lines))
                break


@session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite."""
    session.install(".")
    session.install("pytest", "pytest-cov", "pytest-mock")
    session.run("pytest", *session.posargs)


@session(name="pre-commit", python=python_versions[0])
def precommit(session: Session) -> None:
    args = session.posargs or [
        "run",
        "--all-files",
        "--show-diff-on-failure",
    ]
    session.install(".")
    session.install(
        "black",
        "flake8",
        "flake8-bugbear",
        "isort",
        "pre-commit",
    )
    session.run("pre-commit", *args)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(session)


@nox.session(python=python_versions)
def coverage(session: Session) -> None:
    """Generate coverage data."""
    args = session.posargs or ["report"]
    session.install("coverage[toml]")

    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", *args)


@nox.session(name="docs-build", python=python_versions[0])
def docs_build(session: Session) -> None:
    """Build the docs with mkdocs."""
    args = session.posargs or ["docs", "docs/_build"]

    session.install(".")
    session.install(session, groups=["docs"], root=True)
    session.run("mkdocs", "build", *args)


#
# @nox.session(name="docs-deploy", python=python_versions[0])
# def docs_deploy(session: Session) -> None:
#     """Build the docs with mkdocs."""
#     args = session.posargs
#     install(session, groups=["docs"], root=True)
#     session.run("mkdocs", "gh-deploy", *args)
