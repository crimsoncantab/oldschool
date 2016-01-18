#include "db.h"
#include "asst4.h"

/*
 * asst4_add_recover --
 *	Recovery function for add.
 *
 * PUBLIC: int asst4_add_recover
 * PUBLIC:   __P((DB_ENV *, DBT *, DB_LSN *, db_recops));
 */
int
asst4_add_recover(DB_ENV *dbenv, DBT * dbtp, DB_LSN * lsnp, db_recops op) {
	asst4_add_args *argp;
	int cmp_n, cmp_p, modified, ret;

#ifdef DEBUG_RECOVER
	(void) asst4_add_print(dbenv, dbtp, lsnp, op);
#endif
	argp = NULL;
	if ((ret = asst4_add_read(dbenv, dbtp->data, &argp)) != 0)
		goto out;

	modified = 0;
	cmp_n = 0;
	cmp_p = 0;
	int fd = open_db();
	//get the record
	record rec;
	record * new;
	my_seek(fd, OFFSET(argp->key), SEEK_SET);
	my_read(fd, &rec, sizeof (record));

	//figure out what to do
	cmp_p = log_compare(&rec.lsn, &argp->lsn);
	cmp_n = log_compare(lsnp, &rec.lsn);
	if (cmp_p == 0 && DB_REDO(op)) {
		new = (record *) argp->new.data;
		new->exists = 'Y';
		new->lsn = *lsnp;
		my_seek(fd, -sizeof (record), SEEK_CUR);
		my_write(fd, new, sizeof (record));
		modified = 1;
	} else if (cmp_n == 0 && DB_UNDO(op)) {
		rec.exists = '\0';
		rec.lsn = argp->lsn;
		my_seek(fd, -sizeof (record), SEEK_CUR);
		my_write(fd, &rec, sizeof (record));
		modified = 1;
	}

	/* Allow for following LSN pointers through a transaction. */
	*lsnp = argp->prev_lsn;
	ret = 0;

out:
	if (argp != NULL)
		free(argp);

	close_db(fd);
	return (ret);
}

/*
 * asst4_del_recover --
 *	Recovery function for del.
 *
 * PUBLIC: int asst4_del_recover
 * PUBLIC:   __P((DB_ENV *, DBT *, DB_LSN *, db_recops));
 */
int asst4_del_recover(DB_ENV *dbenv, DBT *dbtp, DB_LSN *lsnp, db_recops op) {
	asst4_del_args *argp;
	int cmp_n, cmp_p, modified, ret;

#ifdef DEBUG_RECOVER
	(void) asst4_del_print(dbenv, dbtp, lsnp, op);
#endif
	argp = NULL;
	if ((ret = asst4_del_read(dbenv, dbtp->data, &argp)) != 0)
		goto out;

	modified = 0;
	cmp_n = 0;
	cmp_p = 0;
	int fd = open_db();
	//get the record
	record rec;
	record * old;
	my_seek(fd, OFFSET(argp->key), SEEK_SET);
	my_read(fd, &rec, sizeof (record));

	//figure out what to do
	cmp_p = log_compare(&rec.lsn, &argp->lsn);
	cmp_n = log_compare(lsnp, &rec.lsn);
	if (cmp_p == 0 && DB_REDO(op)) {
		rec.exists = '\0';
		rec.lsn = *lsnp;
		my_seek(fd, -sizeof (record), SEEK_CUR);
		my_write(fd, &rec, sizeof (record));
		modified = 1;
	} else if (cmp_n == 0 && DB_UNDO(op)) {
		old = (record *) argp->old.data;
		old->exists = 'Y';
		old->lsn = argp->lsn;
		my_seek(fd, -sizeof (record), SEEK_CUR);
		my_write(fd, old, sizeof (record));
		modified = 1;
	}

	/* Allow for following LSN pointers through a transaction. */
	*lsnp = argp->prev_lsn;
	ret = 0;

out:
	if (argp != NULL)
		free(argp);

	close_db(fd);
	return (ret);
}

/*
 * asst4_mod_recover --
 *	Recovery function for mod.
 *
 * PUBLIC: int asst4_mod_recover
 * PUBLIC:   __P((DB_ENV *, DBT *, DB_LSN *, db_recops));
 */
int asst4_mod_recover(DB_ENV *dbenv, DBT *dbtp, DB_LSN *lsnp, db_recops op) {
	asst4_mod_args *argp;
	int cmp_n, cmp_p, modified, ret;

#ifdef DEBUG_RECOVER
	(void) asst4_mod_print(dbenv, dbtp, lsnp, op);
#endif
	argp = NULL;
	if ((ret = asst4_mod_read(dbenv, dbtp->data, &argp)) != 0)
		goto out;

	modified = 0;
	cmp_n = 0;
	cmp_p = 0;
	int fd = open_db();
	//get the record
	record rec;
	record * correct;
	my_seek(fd, OFFSET(argp->key), SEEK_SET);
	my_read(fd, &rec, sizeof (record));

	//figure out what to do
	cmp_p = log_compare(&rec.lsn, &argp->lsn);
	cmp_n = log_compare(lsnp, &rec.lsn);
	if (cmp_p == 0 && DB_REDO(op)) {
		correct = (record *) argp->new.data;
		correct->exists = 'Y';
		correct->lsn = *lsnp;
		my_seek(fd, -sizeof (record), SEEK_CUR);
		my_write(fd, correct, sizeof (record));
		modified = 1;
	} else if (cmp_n == 0 && DB_UNDO(op)) {
		correct = (record *) argp->old.data;
		correct->exists = 'Y';
		correct->lsn = argp->lsn;
		my_seek(fd, -sizeof (record), SEEK_CUR);
		my_write(fd, correct, sizeof (record));
		modified = 1;
	}

	/* Allow for following LSN pointers through a transaction. */
	*lsnp = argp->prev_lsn;
	ret = 0;

out:
	if (argp != NULL)
		free(argp);

	close_db(fd);
	return (ret);
}

