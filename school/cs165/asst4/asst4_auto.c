/* Do not edit: automatically built by gen_rec.awk. */

#include "db_config.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include "db.h"
#include "db_int.h"
#include "dbinc/db_swap.h"
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <db.h>

#include "asst4.h"
/*
 * PUBLIC: int asst4_add_read __P((DB_ENV *, void *, asst4_add_args **));
 */
int
asst4_add_read(dbenv, recbuf, argpp)
	DB_ENV *dbenv;
	void *recbuf;
	asst4_add_args **argpp;
{
	asst4_add_args *argp;
	u_int8_t *bp;
	ENV *env;

	env = dbenv->env;

	if ((argp = malloc(sizeof(asst4_add_args) + sizeof(DB_TXN))) == NULL)
		return (ENOMEM);
	bp = recbuf;
	argp->txnp = (DB_TXN *)&argp[1];
	memset(argp->txnp, 0, sizeof(DB_TXN));

	LOGCOPY_32(env, &argp->type, bp);
	bp += sizeof(argp->type);

	LOGCOPY_32(env, &argp->txnp->txnid, bp);
	bp += sizeof(argp->txnp->txnid);

	LOGCOPY_TOLSN(env, &argp->prev_lsn, bp);
	bp += sizeof(DB_LSN);

	LOGCOPY_TOLSN(env, &argp->lsn, bp);
	bp += sizeof(DB_LSN);

	LOGCOPY_32(env, &argp->key, bp);
	bp += sizeof(argp->key);

	memset(&argp->new, 0, sizeof(argp->new));
	LOGCOPY_32(env,&argp->new.size, bp);
	bp += sizeof(u_int32_t);
	argp->new.data = bp;
	bp += argp->new.size;

	*argpp = argp;
	return (0);
}

/*
 * PUBLIC: int asst4_add_log __P((DB_ENV *, DB_TXN *, DB_LSN *,
 * PUBLIC:     u_int32_t, DB_LSN *, u_int32_t, const DBT *));
 */
int
asst4_add_log(dbenv, txnp, ret_lsnp, flags,
    lsn, key, new)
	DB_ENV *dbenv;
	DB_TXN *txnp;
	DB_LSN *ret_lsnp;
	u_int32_t flags;
	DB_LSN * lsn;
	u_int32_t key;
	const DBT *new;
{
	DBT logrec;
	DB_LSN *lsnp, null_lsn, *rlsnp;
	ENV *env;
	u_int32_t zero, rectype, txn_num;
	u_int npad;
	u_int8_t *bp;
	int ret;

	env = dbenv->env;
	rlsnp = ret_lsnp;
	rectype = DB_asst4_add;
	npad = 0;
	ret = 0;

	if (txnp == NULL) {
		txn_num = 0;
		lsnp = &null_lsn;
		null_lsn.file = null_lsn.offset = 0;
	} else {
		/*
		 * We need to assign begin_lsn while holding region mutex.
		 * That assignment is done inside the DbEnv->log_put call,
		 * so pass in the appropriate memory location to be filled
		 * in by the log_put code.
		 */
		DB_SET_TXN_LSNP(txnp, &rlsnp, &lsnp);
		txn_num = txnp->txnid;
	}

	logrec.size = sizeof(rectype) + sizeof(txn_num) + sizeof(DB_LSN)
	    + sizeof(*lsn)
	    + sizeof(u_int32_t)
	    + sizeof(u_int32_t) + (new == NULL ? 0 : new->size);
	if ((logrec.data = malloc(logrec.size)) == NULL)
		return (ENOMEM);
	bp = logrec.data;

	if (npad > 0)
		memset((u_int8_t *)logrec.data + logrec.size - npad, 0, npad);

	bp = logrec.data;

	LOGCOPY_32(env, bp, &rectype);
	bp += sizeof(rectype);

	LOGCOPY_32(env, bp, &txn_num);
	bp += sizeof(txn_num);

	LOGCOPY_FROMLSN(env, bp, lsnp);
	bp += sizeof(DB_LSN);

	if (lsn != NULL)
		LOGCOPY_FROMLSN(env, bp, lsn);
	else
		memset(bp, 0, sizeof(*lsn));
	bp += sizeof(*lsn);

	LOGCOPY_32(env, bp, &key);
	bp += sizeof(key);

	if (new == NULL) {
		zero = 0;
		LOGCOPY_32(env, bp, &zero);
		bp += sizeof(u_int32_t);
	} else {
		LOGCOPY_32(env, bp, &new->size);
		bp += sizeof(new->size);
		memcpy(bp, new->data, new->size);
		bp += new->size;
	}

	if ((ret = dbenv->log_put(dbenv, rlsnp, (DBT *)&logrec,
	    flags | DB_LOG_NOCOPY)) == 0 && txnp != NULL) {
		*lsnp = *rlsnp;
		if (rlsnp != ret_lsnp)
			 *ret_lsnp = *rlsnp;
	}
#ifdef LOG_DIAGNOSTIC
	if (ret != 0)
		(void)asst4_add_print(dbenv,
		    (DBT *)&logrec, ret_lsnp, DB_TXN_PRINT);
#endif

	free(logrec.data);
	return (ret);
}

/*
 * PUBLIC: int asst4_del_read __P((DB_ENV *, void *, asst4_del_args **));
 */
int
asst4_del_read(dbenv, recbuf, argpp)
	DB_ENV *dbenv;
	void *recbuf;
	asst4_del_args **argpp;
{
	asst4_del_args *argp;
	u_int8_t *bp;
	ENV *env;

	env = dbenv->env;

	if ((argp = malloc(sizeof(asst4_del_args) + sizeof(DB_TXN))) == NULL)
		return (ENOMEM);
	bp = recbuf;
	argp->txnp = (DB_TXN *)&argp[1];
	memset(argp->txnp, 0, sizeof(DB_TXN));

	LOGCOPY_32(env, &argp->type, bp);
	bp += sizeof(argp->type);

	LOGCOPY_32(env, &argp->txnp->txnid, bp);
	bp += sizeof(argp->txnp->txnid);

	LOGCOPY_TOLSN(env, &argp->prev_lsn, bp);
	bp += sizeof(DB_LSN);

	LOGCOPY_TOLSN(env, &argp->lsn, bp);
	bp += sizeof(DB_LSN);

	LOGCOPY_32(env, &argp->key, bp);
	bp += sizeof(argp->key);

	memset(&argp->old, 0, sizeof(argp->old));
	LOGCOPY_32(env,&argp->old.size, bp);
	bp += sizeof(u_int32_t);
	argp->old.data = bp;
	bp += argp->old.size;

	*argpp = argp;
	return (0);
}

/*
 * PUBLIC: int asst4_del_log __P((DB_ENV *, DB_TXN *, DB_LSN *,
 * PUBLIC:     u_int32_t, DB_LSN *, u_int32_t, const DBT *));
 */
int
asst4_del_log(dbenv, txnp, ret_lsnp, flags,
    lsn, key, old)
	DB_ENV *dbenv;
	DB_TXN *txnp;
	DB_LSN *ret_lsnp;
	u_int32_t flags;
	DB_LSN * lsn;
	u_int32_t key;
	const DBT *old;
{
	DBT logrec;
	DB_LSN *lsnp, null_lsn, *rlsnp;
	ENV *env;
	u_int32_t zero, rectype, txn_num;
	u_int npad;
	u_int8_t *bp;
	int ret;

	env = dbenv->env;
	rlsnp = ret_lsnp;
	rectype = DB_asst4_del;
	npad = 0;
	ret = 0;

	if (txnp == NULL) {
		txn_num = 0;
		lsnp = &null_lsn;
		null_lsn.file = null_lsn.offset = 0;
	} else {
		/*
		 * We need to assign begin_lsn while holding region mutex.
		 * That assignment is done inside the DbEnv->log_put call,
		 * so pass in the appropriate memory location to be filled
		 * in by the log_put code.
		 */
		DB_SET_TXN_LSNP(txnp, &rlsnp, &lsnp);
		txn_num = txnp->txnid;
	}

	logrec.size = sizeof(rectype) + sizeof(txn_num) + sizeof(DB_LSN)
	    + sizeof(*lsn)
	    + sizeof(u_int32_t)
	    + sizeof(u_int32_t) + (old == NULL ? 0 : old->size);
	if ((logrec.data = malloc(logrec.size)) == NULL)
		return (ENOMEM);
	bp = logrec.data;

	if (npad > 0)
		memset((u_int8_t *)logrec.data + logrec.size - npad, 0, npad);

	bp = logrec.data;

	LOGCOPY_32(env, bp, &rectype);
	bp += sizeof(rectype);

	LOGCOPY_32(env, bp, &txn_num);
	bp += sizeof(txn_num);

	LOGCOPY_FROMLSN(env, bp, lsnp);
	bp += sizeof(DB_LSN);

	if (lsn != NULL)
		LOGCOPY_FROMLSN(env, bp, lsn);
	else
		memset(bp, 0, sizeof(*lsn));
	bp += sizeof(*lsn);

	LOGCOPY_32(env, bp, &key);
	bp += sizeof(key);

	if (old == NULL) {
		zero = 0;
		LOGCOPY_32(env, bp, &zero);
		bp += sizeof(u_int32_t);
	} else {
		LOGCOPY_32(env, bp, &old->size);
		bp += sizeof(old->size);
		memcpy(bp, old->data, old->size);
		bp += old->size;
	}

	if ((ret = dbenv->log_put(dbenv, rlsnp, (DBT *)&logrec,
	    flags | DB_LOG_NOCOPY)) == 0 && txnp != NULL) {
		*lsnp = *rlsnp;
		if (rlsnp != ret_lsnp)
			 *ret_lsnp = *rlsnp;
	}
#ifdef LOG_DIAGNOSTIC
	if (ret != 0)
		(void)asst4_del_print(dbenv,
		    (DBT *)&logrec, ret_lsnp, DB_TXN_PRINT);
#endif

	free(logrec.data);
	return (ret);
}

/*
 * PUBLIC: int asst4_mod_read __P((DB_ENV *, void *, asst4_mod_args **));
 */
int
asst4_mod_read(dbenv, recbuf, argpp)
	DB_ENV *dbenv;
	void *recbuf;
	asst4_mod_args **argpp;
{
	asst4_mod_args *argp;
	u_int8_t *bp;
	ENV *env;

	env = dbenv->env;

	if ((argp = malloc(sizeof(asst4_mod_args) + sizeof(DB_TXN))) == NULL)
		return (ENOMEM);
	bp = recbuf;
	argp->txnp = (DB_TXN *)&argp[1];
	memset(argp->txnp, 0, sizeof(DB_TXN));

	LOGCOPY_32(env, &argp->type, bp);
	bp += sizeof(argp->type);

	LOGCOPY_32(env, &argp->txnp->txnid, bp);
	bp += sizeof(argp->txnp->txnid);

	LOGCOPY_TOLSN(env, &argp->prev_lsn, bp);
	bp += sizeof(DB_LSN);

	LOGCOPY_TOLSN(env, &argp->lsn, bp);
	bp += sizeof(DB_LSN);

	LOGCOPY_32(env, &argp->key, bp);
	bp += sizeof(argp->key);

	memset(&argp->old, 0, sizeof(argp->old));
	LOGCOPY_32(env,&argp->old.size, bp);
	bp += sizeof(u_int32_t);
	argp->old.data = bp;
	bp += argp->old.size;

	memset(&argp->new, 0, sizeof(argp->new));
	LOGCOPY_32(env,&argp->new.size, bp);
	bp += sizeof(u_int32_t);
	argp->new.data = bp;
	bp += argp->new.size;

	*argpp = argp;
	return (0);
}

/*
 * PUBLIC: int asst4_mod_log __P((DB_ENV *, DB_TXN *, DB_LSN *,
 * PUBLIC:     u_int32_t, DB_LSN *, u_int32_t, const DBT *, const DBT *));
 */
int
asst4_mod_log(dbenv, txnp, ret_lsnp, flags,
    lsn, key, old, new)
	DB_ENV *dbenv;
	DB_TXN *txnp;
	DB_LSN *ret_lsnp;
	u_int32_t flags;
	DB_LSN * lsn;
	u_int32_t key;
	const DBT *old;
	const DBT *new;
{
	DBT logrec;
	DB_LSN *lsnp, null_lsn, *rlsnp;
	ENV *env;
	u_int32_t zero, rectype, txn_num;
	u_int npad;
	u_int8_t *bp;
	int ret;

	env = dbenv->env;
	rlsnp = ret_lsnp;
	rectype = DB_asst4_mod;
	npad = 0;
	ret = 0;

	if (txnp == NULL) {
		txn_num = 0;
		lsnp = &null_lsn;
		null_lsn.file = null_lsn.offset = 0;
	} else {
		/*
		 * We need to assign begin_lsn while holding region mutex.
		 * That assignment is done inside the DbEnv->log_put call,
		 * so pass in the appropriate memory location to be filled
		 * in by the log_put code.
		 */
		DB_SET_TXN_LSNP(txnp, &rlsnp, &lsnp);
		txn_num = txnp->txnid;
	}

	logrec.size = sizeof(rectype) + sizeof(txn_num) + sizeof(DB_LSN)
	    + sizeof(*lsn)
	    + sizeof(u_int32_t)
	    + sizeof(u_int32_t) + (old == NULL ? 0 : old->size)
	    + sizeof(u_int32_t) + (new == NULL ? 0 : new->size);
	if ((logrec.data = malloc(logrec.size)) == NULL)
		return (ENOMEM);
	bp = logrec.data;

	if (npad > 0)
		memset((u_int8_t *)logrec.data + logrec.size - npad, 0, npad);

	bp = logrec.data;

	LOGCOPY_32(env, bp, &rectype);
	bp += sizeof(rectype);

	LOGCOPY_32(env, bp, &txn_num);
	bp += sizeof(txn_num);

	LOGCOPY_FROMLSN(env, bp, lsnp);
	bp += sizeof(DB_LSN);

	if (lsn != NULL)
		LOGCOPY_FROMLSN(env, bp, lsn);
	else
		memset(bp, 0, sizeof(*lsn));
	bp += sizeof(*lsn);

	LOGCOPY_32(env, bp, &key);
	bp += sizeof(key);

	if (old == NULL) {
		zero = 0;
		LOGCOPY_32(env, bp, &zero);
		bp += sizeof(u_int32_t);
	} else {
		LOGCOPY_32(env, bp, &old->size);
		bp += sizeof(old->size);
		memcpy(bp, old->data, old->size);
		bp += old->size;
	}

	if (new == NULL) {
		zero = 0;
		LOGCOPY_32(env, bp, &zero);
		bp += sizeof(u_int32_t);
	} else {
		LOGCOPY_32(env, bp, &new->size);
		bp += sizeof(new->size);
		memcpy(bp, new->data, new->size);
		bp += new->size;
	}

	if ((ret = dbenv->log_put(dbenv, rlsnp, (DBT *)&logrec,
	    flags | DB_LOG_NOCOPY)) == 0 && txnp != NULL) {
		*lsnp = *rlsnp;
		if (rlsnp != ret_lsnp)
			 *ret_lsnp = *rlsnp;
	}
#ifdef LOG_DIAGNOSTIC
	if (ret != 0)
		(void)asst4_mod_print(dbenv,
		    (DBT *)&logrec, ret_lsnp, DB_TXN_PRINT);
#endif

	free(logrec.data);
	return (ret);
}

