/* Do not edit: automatically built by gen_rec.awk. */

#include "db_config.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include "db.h"
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <db.h>

#include "asst4.h"
/*
 * PUBLIC: int asst4_add_print __P((DB_ENV *, DBT *, DB_LSN *, db_recops));
 */
int
asst4_add_print(dbenv, dbtp, lsnp, notused2)
	DB_ENV *dbenv;
	DBT *dbtp;
	DB_LSN *lsnp;
	db_recops notused2;
{
	asst4_add_args *argp;
	int asst4_add_read __P((DB_ENV *, void *, asst4_add_args **));
	u_int32_t i;
	int ch;
	int ret;

	notused2 = DB_TXN_PRINT;

	if ((ret = asst4_add_read(dbenv, dbtp->data, &argp)) != 0)
		return (ret);
	(void)printf(
    "[%lu][%lu]asst4_add%s: rec: %lu txnp %lx prevlsn [%lu][%lu]\n",
	    (u_long)lsnp->file, (u_long)lsnp->offset,
	    (argp->type & DB_debug_FLAG) ? "_debug" : "",
	    (u_long)argp->type,
	    (u_long)argp->txnp->txnid,
	    (u_long)argp->prev_lsn.file, (u_long)argp->prev_lsn.offset);
	(void)printf("\tlsn: [%lu][%lu]\n",
	    (u_long)argp->lsn.file, (u_long)argp->lsn.offset);
	(void)printf("\tkey: %lu\n", (u_long)argp->key);
	(void)printf("\tnew: ");
	for (i = 0; i < argp->new.size; i++) {
		ch = ((u_int8_t *)argp->new.data)[i];
		printf(isprint(ch) || ch == 0x0a ? "%c" : "%#x ", ch);
	}
	(void)printf("\n");
	(void)printf("\n");
	free(argp);
	return (0);
}

/*
 * PUBLIC: int asst4_del_print __P((DB_ENV *, DBT *, DB_LSN *, db_recops));
 */
int
asst4_del_print(dbenv, dbtp, lsnp, notused2)
	DB_ENV *dbenv;
	DBT *dbtp;
	DB_LSN *lsnp;
	db_recops notused2;
{
	asst4_del_args *argp;
	int asst4_del_read __P((DB_ENV *, void *, asst4_del_args **));
	u_int32_t i;
	int ch;
	int ret;

	notused2 = DB_TXN_PRINT;

	if ((ret = asst4_del_read(dbenv, dbtp->data, &argp)) != 0)
		return (ret);
	(void)printf(
    "[%lu][%lu]asst4_del%s: rec: %lu txnp %lx prevlsn [%lu][%lu]\n",
	    (u_long)lsnp->file, (u_long)lsnp->offset,
	    (argp->type & DB_debug_FLAG) ? "_debug" : "",
	    (u_long)argp->type,
	    (u_long)argp->txnp->txnid,
	    (u_long)argp->prev_lsn.file, (u_long)argp->prev_lsn.offset);
	(void)printf("\tlsn: [%lu][%lu]\n",
	    (u_long)argp->lsn.file, (u_long)argp->lsn.offset);
	(void)printf("\tkey: %lu\n", (u_long)argp->key);
	(void)printf("\told: ");
	for (i = 0; i < argp->old.size; i++) {
		ch = ((u_int8_t *)argp->old.data)[i];
		printf(isprint(ch) || ch == 0x0a ? "%c" : "%#x ", ch);
	}
	(void)printf("\n");
	(void)printf("\n");
	free(argp);
	return (0);
}

/*
 * PUBLIC: int asst4_mod_print __P((DB_ENV *, DBT *, DB_LSN *, db_recops));
 */
int
asst4_mod_print(dbenv, dbtp, lsnp, notused2)
	DB_ENV *dbenv;
	DBT *dbtp;
	DB_LSN *lsnp;
	db_recops notused2;
{
	asst4_mod_args *argp;
	int asst4_mod_read __P((DB_ENV *, void *, asst4_mod_args **));
	u_int32_t i;
	int ch;
	int ret;

	notused2 = DB_TXN_PRINT;

	if ((ret = asst4_mod_read(dbenv, dbtp->data, &argp)) != 0)
		return (ret);
	(void)printf(
    "[%lu][%lu]asst4_mod%s: rec: %lu txnp %lx prevlsn [%lu][%lu]\n",
	    (u_long)lsnp->file, (u_long)lsnp->offset,
	    (argp->type & DB_debug_FLAG) ? "_debug" : "",
	    (u_long)argp->type,
	    (u_long)argp->txnp->txnid,
	    (u_long)argp->prev_lsn.file, (u_long)argp->prev_lsn.offset);
	(void)printf("\tlsn: [%lu][%lu]\n",
	    (u_long)argp->lsn.file, (u_long)argp->lsn.offset);
	(void)printf("\tkey: %lu\n", (u_long)argp->key);
	(void)printf("\told: ");
	for (i = 0; i < argp->old.size; i++) {
		ch = ((u_int8_t *)argp->old.data)[i];
		printf(isprint(ch) || ch == 0x0a ? "%c" : "%#x ", ch);
	}
	(void)printf("\n");
	(void)printf("\tnew: ");
	for (i = 0; i < argp->new.size; i++) {
		ch = ((u_int8_t *)argp->new.data)[i];
		printf(isprint(ch) || ch == 0x0a ? "%c" : "%#x ", ch);
	}
	(void)printf("\n");
	(void)printf("\n");
	free(argp);
	return (0);
}

/*
 * PUBLIC: int asst4_init_print __P((DB_ENV *, DB_DISTAB *));
 */
int
asst4_init_print(dbenv, dtabp)
	DB_ENV *dbenv;
	DB_DISTAB *dtabp;
{
	int __db_add_recovery __P((DB_ENV *, DB_DISTAB *,
	    int (*)(DB_ENV *, DBT *, DB_LSN *, db_recops), u_int32_t));
	int ret;

	if ((ret = __db_add_recovery(dbenv, dtabp,
	    asst4_add_print, DB_asst4_add)) != 0)
		return (ret);
	if ((ret = __db_add_recovery(dbenv, dtabp,
	    asst4_del_print, DB_asst4_del)) != 0)
		return (ret);
	if ((ret = __db_add_recovery(dbenv, dtabp,
	    asst4_mod_print, DB_asst4_mod)) != 0)
		return (ret);
	return (0);
}
