from pydal import DAL

from DB.tables_Def import create_tables
from common import logger


def init_db(path='sqlite://storage.db'):
    '''Connect to DB'''

    global db
    db = DAL(path)
    db = create_tables(db)
    logger.error('*******' + str(db))
    db.commit()
    return db
try:
    db = init_db()
    logger.error('*******' + str(db))
except Exception as e:
    '''For local debug'''