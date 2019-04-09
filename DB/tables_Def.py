from pydal import Field


class SqlLiteTables:
    t_users = 't_users'
    t_users_msg = 't_users_msg'


def create_tables(db):
    #### USERS TABLE
    db.define_table('t_users',
                    Field('f_full_name', type='string'),
                    Field('username', type='string'),
                    Field('chat_id', type='string'),
                    Field('last_msg_date', type='datetime'),
                    Field('isAdmin', type='boolean', default=False),
                    Field('user_id', type='string'),
                    Field('isActive', type='boolean', default=True))

    ###### USERS MSG TABLE

    db.define_table('t_user_pics',
                    Field('user_id', 'reference t_users'),
                    Field('pic', type='blob'),
                    Field('score', type='string'),
                    Field('tg_msg_id', type='string'))
    return db
