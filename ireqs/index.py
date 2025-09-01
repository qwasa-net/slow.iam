from .db import DB
from .main import parse_args, run
from .processors import DBInsertProcessor, SaveProcessor


def main(config=None):
    config = config or parse_args(with_command="index")
    processors = []
    if config.save_img:
        processors.append(SaveProcessor(config))
    if config.database:
        db = DB(
            connection_string=config.database,
            prefix=config.database_prefix,
        )
        db.init_table()
        processors.append(DBInsertProcessor(config, db))
    run(processors, config)


if __name__ == "__main__":
    main()
