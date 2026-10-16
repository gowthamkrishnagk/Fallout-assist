"""Local administration without a default password."""

from __future__ import annotations

import argparse
import getpass

from falloutassist_next import storage
from falloutassist_next.auth import create


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["create-user"])
    parser.add_argument("email")
    parser.add_argument("--role", choices=["admin", "user"], default="admin")
    args = parser.parse_args()
    storage.initialize()
    password = getpass.getpass("Password (12+ characters): ")
    print(create(args.email, password, args.role))


if __name__ == "__main__":
    main()
