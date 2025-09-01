from .db import DB
from .main import parse_args, run
from .processors import DBInsertProcessor, SaveProcessor


def main(config=None):
    config = config or parse_args(with_command="cut")
    processors = []
    processors.append(SaveProcessor(config))
    run(processors, config)


if __name__ == "__main__":
    main()
