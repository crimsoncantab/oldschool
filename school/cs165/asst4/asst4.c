/*-
 * This file is derived from one in the db-4.8.26 distribution that bears
 * the following copyright and license (ex_apprec.c).
 *
 * See the file LICENSE for redistribution information.
 *
 * Copyright (c) 1996-2009 Oracle.  All rights reserved.
 *
 * $Id$
 */

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/uio.h>

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <db.h>
#include <asst4.h>

static int add_item(DB_ENV *, int, DB_TXN *, u_int32_t, char *, u_int32_t);
static int asst4_dispatch(DB_ENV *, DBT *, DB_LSN *, db_recops);
static int del_item(DB_ENV *, int, DB_TXN *, u_int32_t);
static void dump_db(DB_ENV *, int);
static int mod_item(DB_ENV *, int, DB_TXN *, u_int32_t, char *, u_int32_t);
static int open_env(const char *, FILE *, const char *, DB_ENV **);

#define MAXDATASIZE 128
#define	MYMAXKEY 128

/* Generate random numbers for transactions .*/
#define	RANDOM_INT(min, max) \
(int) (min + (((double) rand()) / ((double)(RAND_MAX) + 1) * (max - min + 1)))

/*
 *  We will maintain a temporary (in-memory) database to hold all the
 *  keys that we have added, so that when we need to look up a key,
 *  we can just pick one randomly.
 */
int main(int argc, char * argv[]) {
	DB *keydb;
	DB_ENV *dbenv;
	DBT keydbt, lookupdbt;
	extern char *optarg;
	DB_TXN *txn;
	int doing_abort, fd, i, key, lookup, nkeys, ntxn, op, ops, ret, val;
	const char *home;
	char ch, strbuf[MAXDATASIZE];
	const char *progname = "asst4"; /* Program name. */

	doing_abort = 0;

	/* Default home -- note that you'll have to create this directory. */
	home = "testdir";

	while ((ch = getopt(argc, argv, "h:")) != EOF)
		switch (ch) {
			case 'h':
				home = optarg;
				break;
			default:
				fprintf(stderr, "usage: %s [-h home]", progname);
				exit(EXIT_FAILURE);
		}

	printf("Set up environment.\n");
	if ((ret = open_env(home, stderr, progname, &dbenv)) != 0)
		return (EXIT_FAILURE);

	/*
	 * Open/Create temporary database.
	 * We will add record numbers to a btree so that we can pick one
	 * randomly, by simply generating a random number between 1 and
	 * the number of records in the tree.
	 */
	if (db_create(&keydb, NULL, 0) != 0)
		return (EXIT_FAILURE);
	if (keydb->set_flags(keydb, DB_RECNUM) != 0)
		return (EXIT_FAILURE);
	if (keydb->open(keydb, NULL, NULL, NULL, DB_BTREE, DB_CREATE, 0644) != 0)
		return (EXIT_FAILURE);
	nkeys = 0;
	memset(&keydbt, 0, sizeof (keydbt));
	keydbt.data = &key;
	keydbt.size = keydbt.ulen = sizeof (key);
	keydbt.flags = DB_DBT_USERMEM;
	memset(&lookupdbt, 0, sizeof (lookupdbt));
	lookupdbt.data = &lookup;
	lookupdbt.size = lookupdbt.ulen = sizeof (lookup);
	lookupdbt.flags = DB_DBT_USERMEM;

	/* OPEN DATABASE. */
	fd = open_db();

	printf("Initial database: \n");
	dump_db(dbenv, fd);

	for (ntxn = 0; ntxn < 10; ntxn++) {

		if ((ret = dbenv->txn_begin(dbenv, NULL, &txn, 0)) != 0) {
			dbenv->err(dbenv, ret, "txn_begin");
			return (EXIT_FAILURE);
		}

		/* Pick 1-3 operations. */
		ops = RANDOM_INT(1, 3);
		for (i = 0; i < ops; i++) {
			/* Pick random operation */
			op = RANDOM_INT(1, 4);

			switch (op) {
					/* Add Item. */
				case 1:
				case 2:
					/* Generate a random key and data values . */
					key = RANDOM_INT(1, MYMAXKEY);
					val = rand();
					snprintf(strbuf, MAXDATASIZE, "strval%d", val);
					printf("%x: Adding key %d with data: %s/%d\n",
							txn->id(txn), key, strbuf, val);
					ret = add_item(dbenv, fd, txn, key, strbuf, val);
					if (ret == 0 || doing_abort) {
						/* Try to add key to temporary database. */
						ret = keydb->put(keydb,
								NULL, &keydbt, &keydbt, DB_NOOVERWRITE);
						if (ret == DB_KEYEXIST) {
							/*
							 * Before we are doing abort, this should only
							 * happen before we implement the real add_item.
							 * However, if we're doing aborts, we may have
							 * aborted an earlier add of this key and so we
							 * should try to add it even if the add_item
							 * call failed.
							 */
							ret = 0;
							break;
						}
						nkeys++;
					}
					break;
					/* Delete Item. */
				case 3:
					/*
					 * Pick random item in tree.
					 * Using the number of keys in the database, generate a random
					 * number and then look that up in our keydb by record number.
					 * Notice that lookup/lookupdbt is the dbt containing the record
					 * number key and key/keydbt will contain the actual key under
					 * which we stored a record int he database.  If the tree is empty;
					 * go on to a different operation.
					 */
					if (nkeys < 1)
						break;
					lookup = RANDOM_INT(1, nkeys);

					/* Use the record number to get the actual key into keydbt. */
					ret = keydb->get(keydb, NULL, &lookupdbt, &keydbt, DB_SET_RECNO);

					/* Until we do aborts, this should never fail. However, once
					 * we start doing aborts, it's possible that nkeys can become
					 * wrong (since it's local and we don't restore it).
					 */
					if (!doing_abort)
						assert(ret == 0);
					else if (ret != 0)
						break;
					printf("%x: Deleting key %d\n", txn->id(txn), key);
					ret = del_item(dbenv, fd, txn, key);
					if (ret == 0) {
						/* Delete item from our key database. */
						ret = keydb->del(keydb, NULL, &keydbt, 0);
						if (!doing_abort)
							assert(ret == 0);
						else if (ret == DB_NOTFOUND)
							ret = 0;
						nkeys--;
					}
					break;
					/* Modify Item. */
				case 4:
					/* Pick random item in tree.  See comments above.  */
					if (nkeys < 1)
						break;
					lookup = RANDOM_INT(1, nkeys);
					ret = keydb->get(keydb,
							NULL, &lookupdbt, &keydbt, DB_SET_RECNO);
					if (!doing_abort)
						assert(ret == 0);
					else if (ret == DB_NOTFOUND)
						ret = 0;
					val = rand();
					snprintf(strbuf, MAXDATASIZE, "strval%d", val);
					printf("%x: Modifying key %d with data: %s/%d\n",
							txn->id(txn), key, strbuf, val);
					ret = mod_item(dbenv, fd, txn, key, strbuf, val);
					break;
			}

			/*
			 * If you follow the assignment, the only error codes
			 * you should see here are those listed below.  If you add
			 * other errors, be sure to add them here.
			 */
			switch (ret) {
				case 0:
					break;
				case ENOENT:
					printf("Key %d not in database\n", key);
					break;
				case EEXIST:
					printf("Key %d already in database\n", key);
					break;
				case ENOSPC:
					printf("Attempt to add key %d failed -- no space available\n", key);
					break;
				default:
					perror(strerror(ret));
					break;
			}
		}

		/*
		 * Initially, we'll commit everything.  After we have
		 * implemented recovery, we can allow aborts.
		 */
		if (RANDOM_INT(0, 1)) {
			/* Now commit the transaction */
			printf("Commit transaction %x.\n", txn->id(txn));
			if ((ret = txn->commit(txn, 0)) != 0) {
				dbenv->err(dbenv, ret, "txn_commit");
				return (EXIT_FAILURE);
			}
		} else {
			doing_abort = 1;

			/* Now abort the transaction */
			printf("Aborting transaction %x.\n", txn->id(txn));
			if ((ret = txn->abort(txn)) != 0) {
				dbenv->err(dbenv, ret, "txn_abort");
				return (EXIT_FAILURE);
			}
		}
	}

	printf("Final database\n");
	dump_db(dbenv, fd);

	/* Close up temporary database. */
	(void) keydb->close(keydb, 0);

	/* CLOSE THE DATABASE */
	close_db(fd);

	/* Close the handle. */
	if ((ret = dbenv->close(dbenv, 0)) != 0) {
		fprintf(stderr, "DB_ENV->close: %s\n", db_strerror(ret));
		return (EXIT_FAILURE);
	}

	/* Opening with DB_RECOVER runs recovery. */
	if ((ret = open_env(home, stderr, progname, &dbenv)) != 0)
		return (EXIT_FAILURE);

	/* OPEN THE DATABASE */
	fd = open_db();
	printf("Database after recovery\n");
	dump_db(dbenv, fd);
	close_db(fd);

	/* Close the handle. */
	if ((ret = dbenv->close(dbenv, 0)) != 0) {
		fprintf(stderr, "DB_ENV->close: %s\n", db_strerror(ret));
		return (EXIT_FAILURE);
	}
	return (EXIT_SUCCESS);
}

int open_env(const char *home, FILE * errfp, const char *progname, DB_ENV ** dbenvp) {
	DB_ENV *dbenv;
	int ret;

	/*
	 * Create an environment object and initialize it for error
	 * reporting.
	 */
	if ((ret = db_env_create(&dbenv, 0)) != 0) {
		fprintf(errfp, "%s: %s\n", progname, db_strerror(ret));
		return (ret);
	}
	dbenv->set_errfile(dbenv, errfp);
	dbenv->set_errpfx(dbenv, progname);

	/* Set up our custom recovery dispatch function. */
	if ((ret = dbenv->set_app_dispatch(dbenv, asst4_dispatch)) != 0) {
		dbenv->err(dbenv, ret, "set_app_dispatch");
		return (ret);
	}
	/*
	 * Open the environment with full transactional support, running
	 * recovery.
	 */
	if ((ret =
			dbenv->open(dbenv, home, DB_CREATE | DB_RECOVER | DB_INIT_LOCK |
			DB_INIT_LOG | DB_INIT_MPOOL | DB_INIT_TXN, 0)) != 0) {
		dbenv->err(dbenv, ret, "environment open: %s", home);
		dbenv->close(dbenv, 0);
		return (ret);
	}

	*dbenvp = dbenv;
	return (0);
}
/* Routines to initialize/create database and perform operations. */

/* ROUTINE TO OPEN DATABASE. */
int open_db() {
	//if file does not exist create it
	int fd;
	u_int32_t i;
	if (access(DB_FILE, F_OK) != 0) {
		//create file
		fd = my_open_s(DB_FILE, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
		//make file 4 KB, set all bytes to null
		record rec;
		memset(&rec, 0, sizeof (record));
		for (i = 1; i <= MYMAXKEY; i++) {
			my_write(fd, (char *) & rec, sizeof (record));
		}
	} else {
		fd = my_open(DB_FILE, O_RDWR);
	}
	my_sync(fd);
	return fd;
}

/* ONE TO CLOSE IT. */
void close_db(int fd) {
	my_close(fd);
}

/* Add, Del, Mod routines. */
static DBT wrap_record(record * rec) {
	DBT dbt;
	memset(&dbt, 0, sizeof (dbt));
	dbt.data = rec;
	dbt.size = sizeof (record);
	return dbt;
}

static int add_item(DB_ENV *dbenv, int fd, DB_TXN * txnid, u_int32_t key, char * strval, u_int32_t intval) {
	record rec;
	int ret;
	if (key > MYMAXKEY) {
		ret = ENOSPC;
	} else {
		//seek to location in database
		my_seek(fd, OFFSET(key), SEEK_SET);

		//get exclusive lock (since we are going to write, getting just a shared
		//lock might cause deadlock if two txns try to upgrade at the same time)

		//read data
		my_read(fd, &rec, sizeof (record));

		//if data is not there, insert the new data
		if (rec.exists == '\0') {
			//init record
			strncpy(rec.str, strval, STRSIZE);
			rec.exists = 'Y';
			rec.val = intval;

			//log change
			DB_LSN old_lsn = rec.lsn;
			DB_LSN new_lsn;
			DBT new_rec = wrap_record(&rec);
			asst4_add_log(dbenv, txnid, &new_lsn, DB_FLUSH, &old_lsn, key, &new_rec);

			//write change
			rec.lsn = new_lsn;
			my_seek(fd, -sizeof (record), SEEK_CUR);
			my_write(fd, &rec, sizeof (record));
			ret = 0;
		} else { //data is there, return error
			ret = EEXIST;
		}
		//release lock
	}
	sleep(1);
	return ret;
}

static int del_item(DB_ENV *dbenv, int fd, DB_TXN * txnid, u_int32_t key) {
	record rec;
	int ret;
	if (key > MYMAXKEY) {
		ret = ENOENT;
	} else {
		//seek to location in database
		my_seek(fd, OFFSET(key), SEEK_SET);

		//get exclusive lock (since we are going to write, getting just a shared
		//lock might cause deadlock if two txns try to upgrade at the same time)

		//read data
		my_read(fd, &rec, sizeof (record));

		//if data is not there, error
		if (rec.exists == '\0') {
			ret = ENOENT;
		} else { //data is there, delete it
			//log change
			DB_LSN old_lsn = rec.lsn;
			DB_LSN new_lsn;
			DBT old_rec = wrap_record(&rec);
			asst4_del_log(dbenv, txnid, &new_lsn, DB_FLUSH, &old_lsn, key, &old_rec);

			//write change
			rec.exists = '\0';
			rec.lsn = new_lsn;
			my_seek(fd, -sizeof (record), SEEK_CUR);
			my_write(fd, &rec, sizeof (record));
			ret = 0;
		}
		//release lock
	}
	sleep(1);
	return ret;
}

static int mod_item(DB_ENV *dbenv, int fd, DB_TXN * txnid, u_int32_t key, char * strval, u_int32_t intval) {
	record rec;
	record rec2;
	int ret;
	if (key > MYMAXKEY) {
		ret = ENOENT;
	} else {
		//seek to location in database
		my_seek(fd, OFFSET(key), SEEK_SET);

		//get exclusive lock (since we are going to write, getting just a shared
		//lock might cause deadlock if two txns try to upgrade at the same time)

		//read data
		my_read(fd, &rec, sizeof (record));

		//if data is not there, error
		if (rec.exists == '\0') {
			ret = ENOENT;
		} else { //data is there, modify it
			//init record
			strncpy(rec2.str, strval, STRSIZE);
			rec2.val = intval;
			rec2.exists = 'Y';

			//log change
			DB_LSN old_lsn = rec.lsn;
			DB_LSN new_lsn;
			DBT old_rec = wrap_record(&rec);
			DBT new_rec = wrap_record(&rec2);
			asst4_mod_log(dbenv, txnid, &new_lsn, DB_FLUSH, &old_lsn, key, &old_rec, &new_rec);

			//write change
			rec2.lsn = new_lsn;
			my_seek(fd, -sizeof (record), SEEK_CUR);
			my_write(fd, &rec2, sizeof (record));
			ret = 0;
		}
		//release lock
	}
	sleep(1);
	return ret;
}

/* A ROUTINE TO DUMP OUT DATABASE */
static void dump_db(DB_ENV * dbenv, int fd) {
	record rec;
	u_int32_t i;
	my_seek(fd, 0L, SEEK_SET);
	for (i = 1; i <= MYMAXKEY; i++) {
		my_read(fd, (char *) & rec, sizeof (record));
		if (rec.exists != '\0') {
			printf("Key:%4d Data:%18.*s%12d  LSN:\t[%d][%d]\n", i, STRSIZE, rec.str, rec.val, rec.lsn.file, rec.lsn.offset);
		}
	}

}

/*
 * Application specific dispatch function.
 */
static int asst4_dispatch(DB_ENV * dbenv, DBT * dbt, DB_LSN * lsn, db_recops op) {
	u_int32_t rectype;

	/* Extract the record type from the log record. */
	memcpy(&rectype, dbt->data, sizeof (rectype));

	switch (rectype) {
		case DB_asst4_add:
			return asst4_add_recover(dbenv, dbt, lsn, op);
			break;
		case DB_asst4_del:
			return asst4_del_recover(dbenv, dbt, lsn, op);
			break;
		default:
			return asst4_mod_recover(dbenv, dbt, lsn, op);
			break;
	}
}

void my_write(int __fd, const void * __buf, size_t __n) {
	size_t rc;
	rc = write(__fd, __buf, __n);
	if (rc < 0) {
		perror("write");
		exit(EXIT_FAILURE);
	}
}

void my_read(int __fd, void * __buf, size_t __nbytes) {
	size_t rc;
	rc = read(__fd, __buf, __nbytes);
	if (rc < 0) {
		perror("read");
		exit(EXIT_FAILURE);
	}
}

void my_sync(int __fd) {
	if (fsync(__fd) < 0) {
		perror("fysnc");
		exit(EXIT_FAILURE);
	}
}

int my_open(char *__file, int __oflag) {
	int fd;
	fd = open(__file, __oflag);
	if (fd < 0) {
		perror("open");
		exit(EXIT_FAILURE);
	}
	return fd;
}

int my_open_s(char *__file, int __oflag, int __sflag) {
	int fd;
	fd = open(__file, __oflag, __sflag);
	if (fd < 0) {
		perror("open");
		exit(EXIT_FAILURE);
	}
	return fd;
}

void my_close(int __fd) {
	if (close(__fd) < 0) {
		perror("close");
		exit(EXIT_FAILURE);
	}
}

off_t my_seek(int __fd, __off_t __offset, int __whence) {
	off_t loc;
	loc = lseek(__fd, __offset, __whence);
	if (loc == -1) {
		perror("lseek");
		exit(EXIT_FAILURE);
	}
	return loc;
}

