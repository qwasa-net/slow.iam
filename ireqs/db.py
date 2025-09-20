import json

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values


class ConnectionParameters(dict):
    def __init__(self, params: dict):
        super().__init__(params)

    @classmethod
    def from_dsn(cls, dsn: str):
        params = psycopg2.extensions.parse_dsn(dsn)
        return cls(params)


class DB:
    def __init__(self, connection_string: str, prefix: str = ""):
        self.conn_params = ConnectionParameters.from_dsn(connection_string)
        self.conn = None
        self.prefix = prefix  # schema.table_prefix
        self.vector_size = 512

    def connect(self):
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(**self.conn_params)
            register_vector(self.conn)

    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def init_table(self):
        self.connect()
        with self.conn.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.prefix}objects (
                    id SERIAL PRIMARY KEY,
                    tag TEXT,
                    data JSONB NULL,
                    vector VECTOR({self.vector_size}) NOT NULL
                );
                """
            )
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.prefix}objects_vector_idx
                ON objects USING ivfflat (vector vector_cosine_ops)
                WITH (lists=250);
                """
            )
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.prefix}objects_tag_idx
                ON objects USING BTREE (tag);
                """
            )
            self.conn.commit()

    def insert_vector(self, tag: str, data: dict, vector: np.ndarray) -> int:
        self.connect()
        datab = json.dumps(data) if data else None
        with self.conn.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self.prefix}objects
                (tag, data, vector)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (tag, datab, vector),
            )
            last_pk = cursor.fetchone()[0]
            self.conn.commit()
            return last_pk

    def insert_vectors(self, data: list[tuple[str, dict | None, np.ndarray]]) -> int:
        self.connect()
        with self.conn.cursor() as cursor:
            execute_values(
                cursor,
                f"""
                INSERT INTO {self.prefix}objects
                (tag, data, vector)
                VALUES %s
                RETURNING id
                """,
                data,
            )
            last_pk = cursor.fetchall()[-1][0] if data else None
            self.conn.commit()
            return last_pk

    def search_similar(self, query_vector: np.ndarray, limit: int = 5) -> list[tuple[int, str, dict, float]]:
        self.connect()
        with self.conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT id, tag, data, vector <=> %s AS distance
                FROM {self.prefix}objects
                ORDER BY distance ASC
                LIMIT %s
                """,
                (query_vector, limit),
            )
            return cursor.fetchall()
