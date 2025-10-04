"""
An argument parser that manages simple logs.

This is intended for use with simple CLI programs that
want to enable users to manage logs with arguments on the
command line, in addition to whatever other arguments
they handle using normal `argparse` semantics.
"""

import logging
from argparse import ArgumentParser, Namespace
from typing import Optional, Sequence


class LoggingArgumentParser(ArgumentParser):
    def __init__(self, logger: logging.Logger, *args, **kwargs):
        """
        An arg parser that accepts --log and sets up logging accordingly.

        This object behaves exactly like an instance of the class
        :py:ref:`argparse.ArgumentParser` except that it also accepts
        arguments `--log` to set the logging level to one of
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" and `--logfile`
        for the name of a file to which to append logs.

        Here is an example of how this class is typically used in a script
        intended to be used from the command line:

        >>> import logging
        >>> from logargparser import LoggingArgumentParser
        >>>
        >>> logger = logging.getLogger(__name__)
        >>>
        >>> def main():
        ...     # Instead of the normal `parser = argparse.ArgumentParser.
        ...     parser = LoggingArgumentParser(logger)
        ...
        ...     # Add as many other arguments as you want.
        ...     parser.add_argument('-o', "--output")
        ...
        ...     args = parser.parse_args()
        ...
        ...     logger.info("The logger was set up for us based on --log and/or --logfile command line args.")
        ...
        ...     if args.output is not None:
        ...         # Do something....
        ...         pass
        >>>
        >>>
        >>> if __name__ == "__main__":
        ...     main()

        Parameters
        ----------
        logger
            A logger, usually one set up with `logger = logging.getLogger(__name__)` in
            the same file as `main`.
        args
            Arguments passed on to the constructor of the superclass `argparse.ArgumentParser`.
        kwargs
            Keyword arguments passed on to the constructor of the superclass `argparse.ArgumentParser`.
        """
        super().__init__(*args, **kwargs)

        self._logger = logger

        self.add_argument(
            "--log",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level.",
            default="WARNING",
        )

        self.add_argument(
            "--logfile",
            type=str,
            help="Optional file path that logs should be appended to. The file will be created if it does not exist.",
        )

    def parse_args(self, args: Optional[Sequence[str]] = None) -> Namespace:
        args = super().parse_args(args)

        level = getattr(logging, args.log)

        logging.basicConfig(level=level, filename=args.logfile)
        self._logger.setLevel(level)

        return args
