/* Do not edit: automatically built by gen_rec.awk. */

#ifndef	asst4_AUTO_H
#define	asst4_AUTO_H
#define	DB_asst4_add	10001
typedef struct _asst4_add_args {
	u_int32_t type;
	DB_TXN *txnp;
	DB_LSN prev_lsn;
	DB_LSN	lsn;
	u_int32_t	key;
	DBT	new;
} asst4_add_args;

#define	DB_asst4_del	10002
typedef struct _asst4_del_args {
	u_int32_t type;
	DB_TXN *txnp;
	DB_LSN prev_lsn;
	DB_LSN	lsn;
	u_int32_t	key;
	DBT	old;
} asst4_del_args;

#define	DB_asst4_mod	10003
typedef struct _asst4_mod_args {
	u_int32_t type;
	DB_TXN *txnp;
	DB_LSN prev_lsn;
	DB_LSN	lsn;
	u_int32_t	key;
	DBT	old;
	DBT	new;
} asst4_mod_args;

#endif
