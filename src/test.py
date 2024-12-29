from utils.db import Database
from utils.queries import *
from datetime import datetime, timezone
from sqlite3 import IntegrityError

db = Database()


with db as conn:
    pass