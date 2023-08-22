"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """NynoFlow."""


if __name__ == "__main__":
    main(prog_name="nynoflow")  # pragma: no cover
