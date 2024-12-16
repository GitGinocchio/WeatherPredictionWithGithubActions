from utils.db import Database, REMOVE_QUOTES, REMOVE_LOCAL_OBS_TIME

db = Database()


with db as conn:
    conn.refactorDatabase(REMOVE_LOCAL_OBS_TIME)