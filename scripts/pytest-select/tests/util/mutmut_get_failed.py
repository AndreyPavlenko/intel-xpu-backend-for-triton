import sys
from mutmut import OK_KILLED
from mutmut.cache import init_db, db_session, Mutant
from pony.orm import count


@init_db
@db_session
def get_failed_count():
    return count(m for m in Mutant if m.status != OK_KILLED)


if __name__ == "__main__":
    sys.exit(get_failed_count())
