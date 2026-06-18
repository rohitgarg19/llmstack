"""``llmstack install`` -- DEPRECATED: use ``init`` then ``configure`` instead.

For backward compatibility, ``install`` runs ``init`` followed by ``configure``.
This command is deprecated and will be removed in a future version.
"""

from __future__ import annotations

from llmstack.commands import configure, init


def run(args: list[str]) -> int:
    """Run init then configure for backward compatibility."""
    # Parse args to separate init flags from configure flags
    init_args = []
    configure_args = []

    # All install args are init args (--force, --current, --next, --external, --print)
    # We need to handle --print specially: it goes to configure, not init
    print_only = False
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--print", "-n"):
            print_only = True
            configure_args.append(arg)
        elif arg in ("-h", "--help"):
            _print_help()
            return 0
        else:
            init_args.append(arg)
        i += 1

    # Run init (unless --print only)
    if not print_only:
        rc = init.run(init_args)
        if rc not in (0, None):
            return rc

    # Run configure
    return configure.run(configure_args)


def _print_help() -> None:
    print(
        "usage: llmstack install [--print] [--current | --next | --external [URL]]\n"
        "\n"
        "DEPRECATED: use `llmstack init` then `llmstack configure` instead.\n"
        "This command runs both for backward compatibility."
    )
