from .db import DB
from .main import parse_args, run
from .processors import DBQueryProcessor, ShowProcessor


def main(config=None):
    config = config or parse_args(with_command="query")
    assert config.database, "database connection string is required"

    processors = []
    db = DB(
        connection_string=config.database,
        prefix=config.database_prefix,
    )
    db.init_table()
    processors.append(DBQueryProcessor(db))

    if config.show:
        processors.append(ShowProcessor(config))

    run(processors, config)

    db.close()


if __name__ == "__main__":
    main()
